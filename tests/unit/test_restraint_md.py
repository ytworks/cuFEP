"""The trajectories from which a restraint is chosen: what each one is, and how they are run."""
import json

import numpy as np
import openmm as mm
import pytest
from conftest import lj_particles

from cufep.sampling import restraint_md
from cufep.sampling.equilibrate import EquilibrationSettings


def tasks_in(directory, n=2, **changes):
    """``n`` short trajectories of the eight particles on the Reference platform."""
    system, positions, box = lj_particles()
    values = dict(production_ns=0.0005, frame_interval_ps=0.2, platform="Reference", precision="double",
                  timestep_fs=4.0, seed=11, md_backend="openmm")
    values.update(changes)
    settings = EquilibrationSettings(**values)
    velocities = np.zeros_like(positions)
    return [restraint_md.TrajectoryTask(
        index=k, path=str(restraint_md.trajectory_file(directory, k)), system_xml=mm.XmlSerializer.serialize(system),
        positions=positions, velocities=velocities, box=box, frame_atoms=list(range(1, 8)), settings=settings,
        timestep_fs=float(settings.timestep_fs),
        made_with={"length_ns": settings.production_ns, "trajectory": k,
                   "start": restraint_md.start_digest(positions, velocities, box)}) for k in range(n)]


def test_a_trajectory_is_one_file_with_its_frames_their_times_and_what_made_it(tmp_path):
    """0.5 ps at 4 fs with a frame every 0.2 ps: 125 steps, frames after 50, 100 and 125 steps."""
    lines = []
    (summary,) = restraint_md.run_trajectories(tasks_in(tmp_path, n=1), [None], lines.append)
    assert summary == {"index": 0, "backend": "openmm", "timestep_fs": 4.0, "frames": 3, "notes": []}
    with np.load(restraint_md.trajectory_file(tmp_path, 0)) as data:
        assert data["frames"].shape == (3, 7, 3) and data["frame_boxes"].shape == (3, 3, 3)
        assert data["frame_atoms"].tolist() == list(range(1, 8))
        assert data["time_ps"].tolist() == pytest.approx([0.2, 0.4, 0.5])
        assert float(data["simulated_ns"]) == pytest.approx(0.0005) and str(data["backend"]) == "openmm"
        assert json.loads(str(data["made_with"]))["trajectory"] == 0 and json.loads(str(data["notes"])) == []
    assert not list(restraint_md.trajectory_file(tmp_path, 0).parent.glob(".*"))        # nothing half written


def test_the_trajectories_differ_by_their_random_numbers_only_wherever_they_run(tmp_path):
    """Trajectory k has the streams of the leg's seed and k: the same frames whether the
    trajectories run one after the other in the job or each in a process of its own, and other
    frames than every other trajectory from the first frame on."""
    in_turn, in_processes = tmp_path / "turn", tmp_path / "processes"
    restraint_md.run_trajectories(tasks_in(in_turn, n=3), [None], lambda line: None)
    restraint_md.run_trajectories(tasks_in(in_processes, n=3), [None, None], lambda line: None)
    one, other = restraint_md.read_trajectories(in_turn, 3), restraint_md.read_trajectories(in_processes, 3)
    assert np.array_equal(one.frames, other.frames) and np.array_equal(one.boxes, other.boxes)
    assert one.trajectory.tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    first = [one.frames[one.trajectory == k][0] for k in range(3)]
    assert not np.allclose(first[0], first[1]) and not np.allclose(first[1], first[2])
    # a leg of another seed has other trajectories; no seed of a stream is zero
    restraint_md.run_trajectories(tasks_in(tmp_path / "seed", n=1, seed=12), [None], lambda line: None)
    assert not np.allclose(restraint_md.read_trajectories(tmp_path / "seed", 1).frames[0], first[0])
    assert all(s > 0 for k in range(8) for s in restraint_md.trajectory_seeds(0, k))
    assert len({restraint_md.trajectory_seeds(5, k) for k in range(8)}) == 8


@pytest.mark.parametrize("timestep_fs,frames,analysed", [(4.0, 300, 200), (3.0, 301, 201)])
def test_the_frames_of_the_first_part_are_left_out_by_their_time(timestep_fs, frames, analysed):
    """3 ns with a frame every 10 ps and the first 1 ns left out: 300 frames at 4 fs, of which 200
    are later than 1 ns. At 3 fs a frame interval is 3333 steps, 9.999 ps, so that there are 301
    frames and 201 of them are later than 1 ns: the time decides, not a count of frames."""
    from cufep.steps import frames_of

    assert frames_of(3000.0, 10.0, timestep_fs) == frames
    times = restraint_md.frame_times_ps(frames, 3.0, 10.0, timestep_fs)
    assert times[-1] == pytest.approx(3000.0) and np.all(np.diff(times) > 0)
    whole = restraint_md.Trajectories(frames=np.zeros((2 * frames, 1, 3)), boxes=np.zeros((2 * frames, 3, 3)),
                                      frame_atoms=np.array([0]), trajectory=np.repeat([0, 1], frames),
                                      time_ps=np.concatenate([times, times]), backends=["openmm"] * 2,
                                      timesteps_fs=[timestep_fs] * 2, notes=[])
    kept = whole.after(1.0)
    assert kept.frames.shape[0] == 2 * analysed and np.all(kept.time_ps > 1000.0)
    assert np.count_nonzero(kept.trajectory == 0) == analysed
    assert whole.after(0.0).frames.shape[0] == 2 * frames


def test_a_trajectory_recovers_as_every_plain_md_does_and_its_file_says_so(tmp_path, other_backend):
    """The backend of the leg runs a trajectory; if it fails, stock OpenMM runs it and the file of
    the trajectory records that backend and the note; without a fallback nothing else runs it."""
    restraint_md.run_trajectories(tasks_in(tmp_path / "plugin", n=1, md_backend="cupotential"), [None],
                                  lambda line: None)
    assert restraint_md.read_trajectories(tmp_path / "plugin", 1).backends == ["cupotential"]
    assert other_backend.contexts == 1
    other_backend.fail = True
    lines = []
    restraint_md.run_trajectories(tasks_in(tmp_path / "failed", n=1, md_backend="cupotential"), [None], lines.append)
    read = restraint_md.read_trajectories(tmp_path / "failed", 1)
    assert read.backends == ["openmm"]
    assert read.notes == ["the MD 0 for the restraint failed on cupotential and was repeated on stock OpenMM"]
    assert lines == ["MD 0 for the restraint failed on cupotential (the plugin stopped); repeating it on stock OpenMM"]
    with pytest.raises(Exception, match="the plugin stopped"):
        restraint_md.run_trajectories(tasks_in(tmp_path / "none", n=1, md_backend="cupotential", fallback="none"),
                                      [None], lambda line: None)
    assert not restraint_md.trajectory_file(tmp_path / "none", 0).exists()


def test_an_unstable_trajectory_is_repeated_at_2_fs_and_its_file_says_so(tmp_path, monkeypatch):
    from cufep.sampling import equilibrate

    run = equilibrate._production

    def production(system, backend, start, frame_atoms, settings, dt_fs, *args, **kwargs):
        if dt_fs > 2.0:
            raise equilibrate.EquilibrationError("energy became non-finite during the pre-equilibration MD")
        return run(system, backend, start, frame_atoms, settings, dt_fs, *args, **kwargs)

    monkeypatch.setattr(equilibrate, "_production", production)
    (summary,) = restraint_md.run_trajectories(tasks_in(tmp_path, n=1), [None], lambda line: None)
    read = restraint_md.read_trajectories(tmp_path, 1)
    assert summary["timestep_fs"] == 2.0 and read.timesteps_fs == [2.0]
    assert read.notes == ["MD 0 for the restraint: time step lowered automatically to 2 fs after an unstable run at "
                          "4 fs"]
    assert read.time_ps.tolist() == pytest.approx([0.2, 0.4, 0.5])


def test_a_process_that_reports_an_error_stops_the_others_and_one_that_dies_is_repeated_on_stock_openmm(
        tmp_path, monkeypatch):
    """An error in the process of a trajectory ends the stage with what it said. A process that
    ends without a word is a failure of the backend: the trajectory runs once more on stock OpenMM
    if the configuration allows that, and its note says so; otherwise the stage ends."""
    broken = tasks_in(tmp_path / "broken", n=2)
    broken[1].system_xml = "<not a system/>"
    with pytest.raises(restraint_md.TrajectoryError, match="MD 1 for the restraint failed"):
        restraint_md.run_trajectories(broken, [None, None], lambda line: None)

    # what the job does when a process has died, without killing one: the task it runs next
    task = tasks_in(tmp_path / "died", n=1, md_backend="cupotential")[0]
    again = restraint_md._on_stock_openmm(task)
    assert again.settings.md_backend == "openmm" and again.index == task.index and again.path == task.path
    assert again.notes == ["the process of MD 0 for the restraint ended without a result on cupotential; it was "
                           "repeated on stock OpenMM"]
    assert restraint_md._on_stock_openmm(again) is None                              # once only
    none = tasks_in(tmp_path / "died", n=1, md_backend="cupotential", fallback="none")[0]
    assert restraint_md._on_stock_openmm(none) is None


def test_a_process_that_dies_is_seen_and_its_trajectory_is_run_again(tmp_path, monkeypatch):
    """A process killed in the middle of its trajectory (as a crash of the plugin ends it): the job
    notices, and runs the trajectory again on stock OpenMM, in a new process."""
    import multiprocessing as mp
    import os
    import signal

    tasks = tasks_in(tmp_path, n=2, md_backend="cupotential", production_ns=0.004)
    started = []
    context = mp.get_context("spawn")
    make = context.Process

    class KilledOnce:
        """A process; the first one of trajectory 0 is killed as soon as it has started."""

        def __init__(self, target=None, args=(), daemon=None):
            self.inner = make(target=target, args=args, daemon=daemon)
            self.task = args[1]

        def start(self):
            self.inner.start()
            started.append((self.task.index, self.task.settings.md_backend))
            if started.count((0, "cupotential")) == 1 and self.task.index == 0 \
                    and self.task.settings.md_backend == "cupotential":
                os.kill(self.inner.pid, signal.SIGKILL)

        def __getattr__(self, name):
            return getattr(self.inner, name)

    class Context:
        Pipe = staticmethod(context.Pipe)
        Process = KilledOnce

    monkeypatch.setattr(restraint_md.mp, "get_context", lambda method: Context)
    lines = []
    # the plugin is not in the processes (no fixture reaches them): "cupotential" cannot run there and
    # stock OpenMM runs trajectory 1; trajectory 0 is the one whose process died
    summaries = restraint_md.run_trajectories(tasks, [None, None], lines.append)
    assert (0, "openmm") in started and started.count((0, "cupotential")) == 1
    assert "the process of MD 0 for the restraint ended without a result; repeating the trajectory on stock OpenMM" \
        in lines
    assert summaries[0]["notes"] == ["the process of MD 0 for the restraint ended without a result on cupotential; "
                                     "it was repeated on stock OpenMM"]
    assert restraint_md.read_trajectories(tmp_path, 2).frames.shape[0] == 2 * 20


class _Pipes:
    """The two ends of a pipe without a process: what is sent into one end is read at the other."""

    def __init__(self):
        self.messages, self.other = [], None

    @classmethod
    def pair(cls):
        parent, child = cls(), cls()
        parent.other, child.other = child, parent
        return parent, child

    def send(self, message):
        self.other.messages.append(message)

    def poll(self, timeout=0.0):
        return bool(self.messages)

    def recv(self):
        return self.messages.pop(0)

    def close(self):
        pass


def test_a_result_sent_as_the_process_ends_is_read_and_a_process_that_does_not_end_is_stopped(tmp_path, monkeypatch):
    """Review of f5a77ce (P3, found by reading). A process that sends its result and ends between the job's look
    at its pipe and its look at the process was taken as one that died: its trajectory was run again on stock
    OpenMM, or the stage ended ("ended without a result"). And a process that has sent its result and does not
    end within the time it is given was forgotten, with its GPU: it is stopped."""
    tasks = tasks_in(tmp_path, n=2)
    made = []

    class Process:
        """Trajectory 0: its result arrives after the look at the pipe, and it is gone at the look at the
        process. Trajectory 1: its result is there at once, and it does not end unless it is killed."""

        def __init__(self, target=None, args=(), daemon=None):
            self.pipe, self.task, self.killed, self.sent = args[0], args[1], False, False
            made.append(self)

        def start(self):
            if self.task.index == 1:
                self.pipe.send(("done", {"index": 1}))

        def is_alive(self):
            if self.task.index == 0:
                if not self.sent:
                    self.pipe.send(("log", "a line of MD 0"))
                    self.pipe.send(("done", {"index": 0}))
                    self.sent = True
                return False
            return not self.killed

        def join(self, timeout=None):
            pass

        def kill(self):
            self.killed = True

    class Context:
        Pipe = staticmethod(_Pipes.pair)

    Context.Process = Process
    monkeypatch.setattr(restraint_md.mp, "get_context", lambda method: Context)
    lines = []
    assert restraint_md.run_trajectories(tasks, [None, None], lines.append) == [{"index": 0}, {"index": 1}]
    assert lines == ["a line of MD 0"]
    assert [process.killed for process in made] == [False, True] and len(made) == 2


def test_a_trajectory_that_was_published_before_its_process_ended_is_not_run_again(tmp_path, monkeypatch):
    """Review of eb6f89f (P1). A process that ends after it has published its trajectory and before it has
    said so was taken for one whose backend failed: the trajectory was run again on stock OpenMM over the
    file, or the stage ended (``backend.fallback: none``). The file is complete when it is there."""
    tasks = tasks_in(tmp_path, n=2, fallback="none")
    started = []

    class Process:
        """It runs its trajectory where it is made, and is gone without a word."""

        def __init__(self, target=None, args=(), daemon=None):
            self.task = args[1]

        def start(self):
            started.append(self.task.index)
            restraint_md.run_trajectory(self.task, None, lambda line: None)

        def is_alive(self):
            return False

        def join(self, timeout=None):
            pass

        def kill(self):
            pass

    class Context:
        Pipe = staticmethod(_Pipes.pair)

    Context.Process = Process
    monkeypatch.setattr(restraint_md.mp, "get_context", lambda method: Context)
    summaries = restraint_md.run_trajectories(tasks, [None, None], lambda line: None)
    assert started == [0, 1]                                    # none was run again
    assert [(s["index"], s["backend"], s["frames"], s["notes"]) for s in summaries] == [(0, "openmm", 3, []),
                                                                                        (1, "openmm", 3, [])]
    # a file of another task (other settings, another start) is not this trajectory
    other = tasks_in(tmp_path, n=1, production_ns=0.001)[0]
    assert restraint_md.published_summary(other) is None and restraint_md.published_summary(tasks[0])["frames"] == 3


def test_a_trajectory_of_a_bilayer_system_has_the_membrane_barostat_and_no_restraints(tmp_path, monkeypatch):
    """The trajectories of a system in a bilayer run with the barostat of the leg, the membrane
    one, at the temperature, pressure and frequency of the settings, with no force that holds
    atoms; and they do not report on the bilayer as its equilibration does."""
    import openmm.unit as unit

    from cufep.sampling import equilibrate

    made = equilibrate._context
    contexts = []

    def keep(*args, **kwargs):
        context, integrator = made(*args, **kwargs)
        contexts.append((context, integrator))
        return context, integrator

    monkeypatch.setattr(equilibrate, "_context", keep)
    lines = []
    tasks = tasks_in(tmp_path, n=1, barostat="membrane", bilayer_report=False, temperature_K=310.0, pressure_bar=2.0,
                     barostat_frequency=5, friction_per_ps=0.5, restrained_atoms=[0, 1], wall_atoms=[2],
                     wall_half_width_nm=1.0)
    restraint_md.run_trajectories(tasks, [None], lines.append)
    (context, integrator), = contexts
    forces = context.getSystem().getForces()
    barostat, = [f for f in forces if isinstance(f, mm.MonteCarloMembraneBarostat)]
    assert barostat.getFrequency() == 5
    assert barostat.getDefaultPressure().value_in_unit(unit.bar) == pytest.approx(2.0)
    assert barostat.getDefaultTemperature().value_in_unit(unit.kelvin) == pytest.approx(310.0)
    assert not [f for f in forces if isinstance(f, (mm.CustomExternalForce, mm.MonteCarloBarostat))]
    assert integrator.getTemperature().value_in_unit(unit.kelvin) == pytest.approx(310.0)
    assert integrator.getFriction().value_in_unit(unit.picosecond ** -1) == pytest.approx(0.5)
    seed, barostat_seed = restraint_md.trajectory_seeds(11, 0)
    assert integrator.getRandomNumberSeed() == seed and barostat.getRandomNumberSeed() == barostat_seed
    assert lines == []


def test_the_length_of_a_stored_trajectory_is_compared_in_whole_steps(tmp_path):
    """Review of 877ab88 (P2). 0.400002 ns at 4 fs is 100000.5 steps, which the MD rounds to 100000: the file it
    wrote holds 0.39999999999999997 ns, and the check of a stored trajectory, which compared the two times with
    half a step of tolerance, refused it for holding less than was asked. Both are counted in steps."""
    from cufep.workflow import WorkflowError, check_restraint_md_file

    made_with = {"length_ns": 0.400002, "timestep_fs": 4.0, "trajectory": 0}
    np.savez(tmp_path / "t.npz", made_with=json.dumps(made_with), simulated_ns=100000 * 4.0 * 1.0e-6, timestep_fs=4.0)
    check_restraint_md_file(tmp_path / "t.npz", 0, made_with, lambda line: None)
    np.savez(tmp_path / "short.npz", made_with=json.dumps(made_with), simulated_ns=99999 * 4.0 * 1.0e-6,
             timestep_fs=4.0)
    with pytest.raises(WorkflowError, match="holds 399.996 ps where 400.002 ps are asked for"):
        check_restraint_md_file(tmp_path / "short.npz", 0, made_with, lambda line: None)


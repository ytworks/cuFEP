"""Replica mixing, MBAR, storage and an inline replica-exchange run on the CPU."""
import numpy as np
import openmm as mm
import pytest

from cufep.alchemy.factory import make_alchemical_system
from cufep.alchemy.schedule import fixed_schedule
from cufep.analysis.mbar import (analyze_leg, assemble, count_round_trips, solve_mbar, stage_boundaries,
                                 statistical_inefficiency)
from cufep.sampling.hrex import ReplicaExchangeSampler, mix_replicas
from cufep.sampling.storage import LegStorage, read_samples
from cufep.sampling.worker import LegSpec


def harmonic_samples(n_iter, k_spring, x0, rng, exchange=True):
    """Exact Gibbs sampling of K harmonic states (beta = 1) with replica mixing."""
    k = len(k_spring)
    sig = 1.0 / np.sqrt(k_spring)
    state_of = np.arange(k)
    u = np.zeros((n_iter, k, k))
    states = np.zeros((n_iter, k), dtype=int)
    xs = np.zeros((n_iter, k))
    for it in range(n_iter):
        x = rng.normal(x0[state_of], sig[state_of])
        u[it] = 0.5 * k_spring[None, :] * (x[:, None] - x0[None, :]) ** 2
        states[it] = state_of
        xs[it] = x
        if exchange:
            state_of, _, _ = mix_replicas(u[it], state_of, rng)
    return u, states, xs


def test_mixing_is_a_permutation_and_accepts_downhill_swaps():
    rng = np.random.default_rng(0)
    u = rng.normal(size=(6, 6))
    new, acc, att = mix_replicas(u, np.arange(6), rng)
    assert sorted(new.tolist()) == list(range(6)) and 0 < acc <= att
    # two replicas each sitting in the other's favourite state must swap
    u2 = np.array([[50.0, 0.0], [0.0, 50.0]])
    new, _, _ = mix_replicas(u2, np.array([0, 1]), rng, n_attempts=20)
    assert new.tolist() == [1, 0]


def test_mixing_depends_only_on_the_seed_and_iteration():
    u = np.random.default_rng(5).normal(size=(8, 8))
    a = mix_replicas(u, np.arange(8), np.random.default_rng([7, 3, 0x5EED]))[0]
    b = mix_replicas(u, np.arange(8), np.random.default_rng([7, 3, 0x5EED]))[0]
    c = mix_replicas(u, np.arange(8), np.random.default_rng([7, 4, 0x5EED]))[0]
    assert a.tolist() == b.tolist() and a.tolist() != c.tolist()


def test_mixing_preserves_each_state_distribution():
    """Detailed balance: with exchange, the samples collected in state k still follow state k."""
    rng = np.random.default_rng(11)
    k_spring = np.array([1.0, 2.0, 4.0, 8.0])
    x0 = np.array([0.0, 0.4, 0.8, 1.2])
    u, states, xs = harmonic_samples(6000, k_spring, x0, rng)
    assert np.mean(states[1:] != states[:-1]) > 0.2                   # replicas really move
    for k in range(4):
        x = xs[states == k]
        se = 1 / np.sqrt(k_spring[k] * len(x))
        assert abs(x.mean() - x0[k]) < 5 * se
        assert x.var() == pytest.approx(1 / k_spring[k], rel=0.1)


def test_mbar_recovers_harmonic_free_energies():
    rng = np.random.default_rng(3)
    k_spring = np.array([1.0, 1.6, 2.6, 4.2, 6.7, 10.7])
    x0 = np.linspace(0.0, 1.0, 6)
    u, states, _ = harmonic_samples(3000, k_spring, x0, rng)
    f, df, overlap = solve_mbar(u, states)
    exact = 0.5 * np.log(k_spring / k_spring[0])                      # -ln Z_k / Z_0
    assert np.all(np.abs(f - exact) < 5 * np.maximum(df, 0.01))
    assert np.allclose(overlap.sum(axis=1), 1.0, atol=1e-6)
    u_kn, n_k = assemble(u, states)
    assert u_kn.shape == (6, 18000) and n_k.tolist() == [3000] * 6
    assert statistical_inefficiency(u, states) < 1.5                  # independent samples
    assert count_round_trips(states) > 5


def test_mbar_start_does_not_change_the_solution_and_jax_is_off():
    """The solve starts from BAR or from given free energies; the answer is that from zeros."""
    import pymbar

    from cufep.analysis.mbar import _pymbar
    _pymbar()
    assert pymbar.mbar_solvers.use_jit is False                       # NumPy solver, see _pymbar
    rng = np.random.default_rng(5)
    k_spring = np.array([1.0, 1.6, 2.6, 4.2, 6.7, 10.7])
    u, states, _ = harmonic_samples(1500, k_spring, np.linspace(0.0, 1.0, 6), rng)
    u_kn, n_k = assemble(u, states)
    zeros = pymbar.MBAR(u_kn - u_kn.min(), n_k, solver_protocol="robust").compute_free_energy_differences()
    f, df, overlap = solve_mbar(u, states)                             # BAR start
    assert np.allclose(f, zeros["Delta_f"][0], atol=1e-8) and np.allclose(df, zeros["dDelta_f"][0], atol=1e-8)
    f2, df2, none = solve_mbar(u, states, initial_f_k=f + 0.3, overlap=False)
    assert none is None and np.allclose(f2, f, atol=1e-8) and np.allclose(df2, df, atol=1e-8)
    # a window without samples cannot start from BAR; the solve still works
    keep = states != 2
    u_kn2 = u.reshape(-1, 6)[keep.reshape(-1)].T
    order = np.argsort(states.reshape(-1)[keep.reshape(-1)], kind="stable")
    from cufep.analysis.mbar import overlap_matrix
    o = overlap_matrix(u_kn2[:, order], np.array([1500, 1500, 0, 1500, 1500, 1500]))
    assert o.shape == (6, 6) and np.all(np.isfinite(o))


def test_analysis_of_a_leg_with_a_window_that_overlaps_no_other_is_bounded():
    """Such a leg has no defined free energy; the solver must still stop (it once ran for an hour)."""
    import time

    rng = np.random.default_rng(7)
    k_spring = np.array([1.0, 1.0, 1.5, 2.2, 3.3, 5.0])
    x0 = np.array([-60.0, 0.0, 0.3, 0.6, 0.9, 1.2])                  # window 0 is far from all others
    u, states, _ = harmonic_samples(1500, k_spring, x0, rng)
    assert np.all(states[:, 0] == 0)                                  # it never exchanges
    t0 = time.perf_counter()
    res = analyze_leg({"u": u, "states": states}, 298.15)
    assert time.perf_counter() - t0 < 60.0
    assert res.min_overlap < 1e-6 and len(res.forward) == 5 and len(res.reverse) == 5
    # the last entries of the forward and reverse series are the estimate from all data
    assert res.forward[-1][1] == res.delta_g and res.reverse[-1][1] == res.delta_g


def test_stage_boundaries():
    s = fixed_schedule(["restrain", "discharge", "vanish"], {"restrain": 2, "discharge": 3, "vanish": 4})
    b = stage_boundaries([x.to_dict() for x in s.states])
    assert b == {"restrain": (0, 2), "discharge": (2, 5), "vanish": (5, 9)}
    s = fixed_schedule(["discharge", "vanish"], {"discharge": 3, "vanish": 4})
    assert stage_boundaries([x.to_dict() for x in s.states]) == {"discharge": (0, 3), "vanish": (3, 7)}


def test_storage_append_truncate_and_checkpoint(tmp_path):
    st = LegStorage(tmp_path / "leg", 3, 3, n_frame_atoms=2)
    for it in range(5):
        st.append(np.full((3, 3), float(it)), np.array([0, 1, 2]), np.ones(3))
        if it % 2 == 0:
            st.append_record(it, {"lrc_du": -1.0 * it, "frame": np.zeros((2, 3), np.float32),
                                  "box": np.eye(3, dtype=np.float32)})
    st.write_checkpoint(np.zeros((3, 4, 3)), np.zeros((3, 4, 3)), np.zeros((3, 3, 3)), [2, 0, 1], 3)
    st.close()
    st = LegStorage(tmp_path / "leg", 3, 3, n_frame_atoms=2)
    ck = st.read_checkpoint()
    assert ck["iteration"] == 3 and ck["state_of_replica"].tolist() == [2, 0, 1]
    st.truncate(ck["iteration"])
    assert st.n_iterations == 3
    st.close()
    data = read_samples(tmp_path / "leg")
    assert data["u"].shape == (3, 3, 3) and data["lrc_iter"].tolist() == [0, 2] and data["frames"].shape[0] == 2
    with pytest.raises(ValueError):
        LegStorage(tmp_path / "leg", 4, 4)


@pytest.fixture(scope="module")
def tiny_leg(water_ligand_system):
    system, pos, box, ligand = water_ligand_system
    alch = make_alchemical_system(system, ligand)
    schedule = fixed_schedule(["discharge", "vanish"], {"discharge": 1, "vanish": 2})
    spec = LegSpec(name="tiny", system_xml=mm.XmlSerializer.serialize(alch.system),
                   states=[s.to_dict() for s in schedule.states], n_atoms=system.getNumParticles(),
                   backend="openmm", platform="CPU", precision="mixed", temperature_K=300.0, pressure_bar=1.0,
                   barostat="isotropic", timestep_fs=2.0, steps_per_iteration=10, seed=5, frame_atoms=list(ligand))
    return spec, pos, box


def test_inline_replica_exchange_runs_resumes_and_analyses(tmp_path, tiny_leg):
    spec, pos, box = tiny_leg
    leg = tmp_path / "hrex"
    with ReplicaExchangeSampler(spec, leg, devices=[None], inline=True, log=lambda m: None) as s:
        s.initialize(pos, box)
        s.run(6, record_stride=2, checkpoint_stride=3)
        assert s.iteration == 6 and sorted(s.state_of_replica.tolist()) == [0, 1, 2, 3]
        final = s.state_of_replica.copy()
    with ReplicaExchangeSampler(spec, leg, devices=[None], inline=True, log=lambda m: None) as s:
        assert s.restore() and s.iteration == 6
        assert s.state_of_replica.tolist() == final.tolist()
        s.run(6, record_stride=2, checkpoint_stride=3)
        s.storage.set_meta(n_equilibration_iterations=2)
    data = read_samples(leg)
    assert data["u"].shape == (12, 4, 4) and np.all(np.isfinite(data["u"][np.arange(12)[:, None], np.arange(4), data["states"]]))
    assert all(sorted(row.tolist()) == [0, 1, 2, 3] for row in data["states"])
    assert data["frames"].shape[1:] == (3, 3) and len(data["frame_iter"]) >= 4
    res = analyze_leg(data, 300.0)
    assert res.n_states == 4 and res.n_iterations_used == 10
    assert np.isfinite(res.delta_g) and set(res.stage_delta_g) == {"discharge", "vanish"}
    # decoupling a water molecule from water costs free energy (hydration free energy ~ -6 kcal/mol)
    assert res.delta_g > 0


def test_windows_are_added_during_the_equilibration_of_replica_exchange(tmp_path, water_ligand_system):
    """A leg that starts with two vanish windows that do not overlap gets a window from its own equilibration."""
    from cufep.alchemy.schedule import AlsSettings, schedule_from_windows
    from cufep.workflow import _equilibration_checks

    system, pos, box, ligand = water_ligand_system
    alch = make_alchemical_system(system, ligand)
    schedule = schedule_from_windows(["discharge", "vanish"], {"discharge": 2, "vanish": 2}, {"discharge": 2, "vanish": 2})
    settings = AlsSettings(overlap_threshold=0.04, candidates=101, max_windows=6)
    messages, restarts = [], 0
    configs = (pos, box, None)
    while True:
        spec = LegSpec(name="tiny", system_xml=mm.XmlSerializer.serialize(alch.system),
                       states=[s.to_dict() for s in schedule.states], n_atoms=system.getNumParticles(),
                       backend="openmm", platform="CPU", precision="mixed", temperature_K=300.0, pressure_bar=1.0,
                       barostat="isotropic", timestep_fs=2.0, steps_per_iteration=5, seed=5, frame_atoms=list(ligand))
        leg = tmp_path / f"hrex{restarts}"
        with ReplicaExchangeSampler(spec, leg, devices=[None], inline=True, log=lambda m: None) as s:
            s.initialize(*configs)
            out = _equilibration_checks(s, schedule, settings, n_equil=24, n_check=12, iteration_ps=0.01,
                                        run_kwargs=dict(checkpoint_stride=12), on_iteration=lambda smp, st: None,
                                        log=messages.append)
            if out is None:
                assert s.iteration == 24                    # the whole equilibration passed without a deficient pair
                break
        new, configs = out
        assert len(new.states) == len(schedule.states) + 1 and new.mode == "adaptive"
        assert configs[0].shape == (len(new.states), system.getNumParticles(), 3) and configs[1].shape[0] == len(new.states)
        assert np.all(np.isfinite(configs[0])) and np.all(np.isfinite(configs[2]))
        assert new.provenance["windows_added_during_equilibration"] == restarts + 1
        schedule, restarts = new, restarts + 1
        if restarts == 3:
            break
    assert restarts >= 1 and len(schedule.windows()["vanish"]) > 2
    checks = schedule.provenance["equilibration_checks"]
    assert any(c["status"] == "inserted" and c["stage"] == "vanish" for c in checks)
    assert all("after_equilibration_ps" in c for c in checks)
    assert any("adding a window" in m for m in messages)

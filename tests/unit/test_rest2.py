"""REST2: the scaled Hamiltonian, its end states and the evaluation of all state energies."""
import math

import numpy as np
import openmm as mm
import openmm.unit as u
import pytest
from conftest import potential, reference_context

from cufep.alchemy.factory import REST2_DQ, REST2_DS, REST2_GROUP, AlchemyError, bonded_pairs, make_alchemical_system
from cufep.backends.base import clone_system
from cufep.sampling.states import (
    AlchemicalState,
    StateEnergyEvaluator,
    rest2_scale_along_leg,
    set_state,
    stage_state,
    validate_states,
)


def scaled_by_hand(system, region, s):
    """The original System with the REST2 rules applied directly to its parameters."""
    a = math.sqrt(s)
    out = clone_system(system)
    bonds = bonded_pairs(out)
    bonded = lambda i, j: (min(i, j), max(i, j)) in bonds                      # noqa: E731
    for force in out.getForces():
        if isinstance(force, mm.NonbondedForce):
            for i in region:
                q, sig, eps = force.getParticleParameters(i)
                force.setParticleParameters(i, q * a, sig, eps * s)
            for k in range(force.getNumExceptions()):
                i, j, qq, sig, eps = force.getExceptionParameters(k)
                n_hot = (i in region) + (j in region)
                force.setExceptionParameters(k, i, j, qq * a ** n_hot, sig, eps * a ** n_hot)
        elif isinstance(force, mm.PeriodicTorsionForce):
            for t in range(force.getNumTorsions()):
                i, j, k, l, n, phase, kval = force.getTorsionParameters(t)
                if all(x in region for x in (i, j, k, l)) and bonded(i, j) and bonded(j, k) and bonded(k, l):
                    force.setTorsionParameters(t, i, j, k, l, n, phase, kval * s)
    return out


@pytest.fixture(scope="module")
def rest(dipeptides_in_water):
    system, top, pos, box, ligand, other = dipeptides_in_water
    hot = other[:12]                                    # ACE and most of ALA of the other peptide
    return make_alchemical_system(system, ligand, rest2=True, rest2_atoms=hot), hot


def test_scale_one_is_the_physical_system(dipeptides_in_water, rest):
    system, top, pos, box, ligand, other = dipeptides_in_water
    alch, hot = rest
    plain = make_alchemical_system(system, ligand)
    e0 = potential(reference_context(system, pos, box))
    ctx = reference_context(alch.system, pos, box)
    set_state(ctx, AlchemicalState(1.0, 1.0, 1.0))
    assert potential(ctx) == pytest.approx(e0, abs=5e-3)
    # every alchemical state without REST2 has the energy of the System built without it
    ctx0 = reference_context(plain.system, pos, box)
    for state in (AlchemicalState(1, 0.4, 1.0), AlchemicalState(1, 0.0, 0.6), AlchemicalState(1, 0.0, 0.0)):
        set_state(ctx, state)
        set_state(ctx0, state)
        assert potential(ctx) == pytest.approx(potential(ctx0), abs=5e-3)
    assert alch.rest2 and alch.rest2_atoms == tuple(hot) and alch.rest2_torsions > 10
    assert not plain.rest2 and REST2_DS not in ctx0.getParameters()


@pytest.mark.parametrize("temperature_scale", [1.5, 2.0, 3.0])
def test_scaled_hamiltonian_is_that_of_the_rest2_rules(dipeptides_in_water, rest, temperature_scale):
    """Coupled ligand: the energy equals the original System with scaled charges, epsilons, pairs and torsions."""
    system, top, pos, box, ligand, other = dipeptides_in_water
    alch, hot = rest
    s = 1.0 / temperature_scale
    region = set(ligand) | set(hot)
    reference = potential(reference_context(scaled_by_hand(system, region, s), pos, box))
    ctx = reference_context(alch.system, pos, box)
    set_state(ctx, AlchemicalState(1.0, 1.0, 1.0, rest2_scale=temperature_scale))
    assert ctx.getParameter(REST2_DS) == pytest.approx(s - 1.0) and ctx.getParameter(REST2_DQ) == pytest.approx(
        math.sqrt(s) - 1.0
    )
    assert potential(ctx) == pytest.approx(reference, abs=5e-3)
    # the torsions inside the region are the only thing in the REST2 force group, and they scale by s
    torsions = potential(ctx, {REST2_GROUP})
    set_state(ctx, AlchemicalState(1.0, 1.0, 1.0))
    assert torsions == pytest.approx(s * potential(ctx, {REST2_GROUP}), rel=1e-9)


def test_only_proper_torsions_inside_the_region_are_scaled(dipeptides_in_water):
    system, top, pos, box, ligand, other = dipeptides_in_water
    alch = make_alchemical_system(system, ligand, rest2=True)
    hot = next(f for f in alch.system.getForces() if isinstance(f, mm.CustomTorsionForce))
    bonds = bonded_pairs(system)
    lig = set(ligand)
    for t in range(hot.getNumTorsions()):
        i, j, k, l, _ = hot.getTorsionParameters(t)
        assert {i, j, k, l} <= lig
        assert all((min(a, b), max(a, b)) in bonds for a, b in ((i, j), (j, k), (k, l)))
    # impropers of the ligand (the central atom is bonded to the other three) keep their force constant
    original = next(f for f in system.getForces() if isinstance(f, mm.PeriodicTorsionForce))
    kept = next(f for f in alch.system.getForces() if isinstance(f, mm.PeriodicTorsionForce))
    n_improper = n_moved = 0
    for t in range(original.getNumTorsions()):
        i, j, k, l, n, phase, kval = original.getTorsionParameters(t)
        k_now = kept.getTorsionParameters(t)[6].value_in_unit(u.kilojoule_per_mole)
        proper = all((min(a, b), max(a, b)) in bonds for a, b in ((i, j), (j, k), (k, l)))
        if {i, j, k, l} <= lig and kval.value_in_unit(u.kilojoule_per_mole) != 0.0:
            if proper:
                assert k_now == 0.0
                n_moved += 1
            else:
                assert k_now == kval.value_in_unit(u.kilojoule_per_mole)
                n_improper += 1
    assert n_moved == hot.getNumTorsions() > 0 and n_improper > 0
    with pytest.raises(AlchemyError):
        make_alchemical_system(system, ligand, rest2_atoms=other[:3])          # hot atoms without rest2
    with pytest.raises(AlchemyError):
        make_alchemical_system(system, ligand, rest2=True, rest2_atoms=ligand[:2])


@pytest.mark.parametrize("with_hot_environment", [True, False])
def test_evaluator_matches_setting_each_state(dipeptides_in_water, with_hot_environment):
    system, top, pos, box, ligand, other = dipeptides_in_water
    alch = make_alchemical_system(system, ligand, rest2=True, rest2_atoms=other[:12] if with_hot_environment else ())
    stages = ["discharge", "vanish"]
    lams = {"discharge": [0.0, 0.3, 0.7, 1.0], "vanish": [0.2, 0.5, 0.8, 1.0]}
    states = [stage_state(st, lam, stages, 2.5) for st in stages for lam in lams[st]]
    validate_states(states)
    assert states[0].rest2_scale == 1.0 and states[-1].rest2_scale == 1.0 and max(s.rest2_scale for s in states) == 2.5
    ctx = reference_context(alch.system, pos, box)
    T, P = 298.15, 1.0
    red, vol = StateEnergyEvaluator(ctx, states, T, P).reduced_potentials(restore=states[2])
    kt = 8.31446261815324e-3 * T
    assert ctx.getParameter(REST2_DS) == pytest.approx(states[2].rest2_energy_scale - 1.0)   # restored
    for k, state in enumerate(states):
        set_state(ctx, state)
        direct = (potential(ctx) + P * vol * 0.0602214076) / kt
        assert red[k] == pytest.approx(direct, abs=2e-3)


def test_rest2_scale_along_a_leg():
    complex_leg = ["restrain", "discharge", "vanish"]
    assert rest2_scale_along_leg(complex_leg, "restrain", 0.0, 3.0) == 1.0
    assert rest2_scale_along_leg(complex_leg, "vanish", 1.0, 3.0) == pytest.approx(1.0)
    assert rest2_scale_along_leg(complex_leg, "discharge", 0.5, 3.0) == pytest.approx(3.0)        # middle of the leg
    assert rest2_scale_along_leg(complex_leg, "restrain", 1.0, 3.0) == pytest.approx(1.0 + 2.0 * 2 / 3)
    assert rest2_scale_along_leg(["discharge", "vanish"], "discharge", 1.0, 2.0) == pytest.approx(2.0)
    assert rest2_scale_along_leg(["discharge", "vanish"], "vanish", 0.5, 2.0) == pytest.approx(1.5)
    assert rest2_scale_along_leg(complex_leg, "vanish", 0.3, 1.0) == 1.0
    assert stage_state("vanish", 0.5) == AlchemicalState(1.0, 0.0, 0.5)
    with pytest.raises(ValueError):
        stage_state("vanish", 0.5, rest2_scale=2.0)                                              # needs the stages
    with pytest.raises(ValueError):
        validate_states([AlchemicalState(1, 1, 1, 1.5), AlchemicalState(1, 0, 0)])               # hot end state
    with pytest.raises(ValueError):
        validate_states([AlchemicalState(1, 1, 1), AlchemicalState(1, 0.5, 1, 0.9), AlchemicalState(1, 0, 0)])


def test_a_system_without_rest2_rejects_scaled_states(water_ligand_system):
    system, pos, box, ligand = water_ligand_system
    alch = make_alchemical_system(system, ligand)
    ctx = reference_context(alch.system, pos, box)
    with pytest.raises(ValueError):
        set_state(ctx, AlchemicalState(1.0, 1.0, 1.0, rest2_scale=2.0))
    with pytest.raises(ValueError):
        StateEnergyEvaluator(ctx, [AlchemicalState(1, 1, 1), AlchemicalState(1, 0.5, 1, 2.0), AlchemicalState(1, 0, 0)],
                             298.15, None)


def test_schedules_carry_the_scale_of_every_window(tmp_path):
    from cufep.alchemy.schedule import Schedule, fixed_schedule, schedule_from_windows

    plain = fixed_schedule(["restrain", "discharge", "vanish"], {"restrain": 2, "discharge": 2, "vanish": 4})
    assert plain.rest2_scale == 1.0 and all(st.rest2_scale == 1.0 for st in plain.states)
    hot = schedule_from_windows(["discharge", "vanish"], {"discharge": 3, "vanish": [0.0, 0.5, 1.0]},
                                {"discharge": 3, "vanish": 3}, rest2_scale=3.0)
    scales = [st.rest2_scale for st in hot.states]
    assert scales == pytest.approx([1.0, 2.0, 3.0, 2.0, 1.0]) and hot.rest2_scale == 3.0
    hot.save(tmp_path / "schedule.yaml")
    again = Schedule.load(tmp_path / "schedule.yaml")
    assert again.rest2_scale == 3.0 and again.states == hot.states
    # a schedule written before REST2 existed loads as unscaled
    import yaml
    d = yaml.safe_load((tmp_path / "schedule.yaml").read_text())
    d.pop("rest2_scale")
    for st in d["states"]:
        st.pop("rest2_scale")
    (tmp_path / "old.yaml").write_text(yaml.safe_dump(d))
    old = Schedule.load(tmp_path / "old.yaml")
    assert old.rest2_scale == 1.0 and all(st.rest2_scale == 1.0 for st in old.states)


def test_precalculation_with_rest2(dipeptides_in_water):
    """Adaptive lambda scheduling runs its windows at the REST2 scale of their place along the leg."""
    from cufep.alchemy.schedule import AlsSettings, adaptive_schedule

    system, top, pos, box, ligand, other = dipeptides_in_water
    alch = make_alchemical_system(system, ligand, rest2=True, rest2_atoms=other[:12])
    integ = mm.LangevinMiddleIntegrator(300.0, 1.0, 0.002)
    integ.setRandomNumberSeed(3)
    ctx = mm.Context(alch.system, integ, mm.Platform.getPlatformByName("CPU"))
    ctx.setPeriodicBoxVectors(*[mm.Vec3(*r) for r in box])
    ctx.setPositions(pos)
    ctx.setVelocitiesToTemperature(300.0, 2)
    settings = AlsSettings(overlap_threshold=0.04, candidates=11, initial_windows={"discharge": 3, "vanish": 3},
                           window_ps=0.2, sample_interval_ps=0.02, max_windows=7, rest2_scale=2.0)
    schedule, configs = adaptive_schedule(ctx, integ, ["discharge", "vanish"], 300.0, 2.0, settings, lambda m: None)
    assert schedule.rest2_scale == 2.0 and len(configs) == len(schedule.states)
    scales = [st.rest2_scale for st in schedule.states]
    assert scales[0] == 1.0 and scales[-1] == 1.0 and max(scales) == pytest.approx(2.0)
    for st, lam, k in zip(schedule.states, schedule.stage_lambda, schedule.stage_of_state):
        assert st.rest2_scale == pytest.approx(rest2_scale_along_leg(schedule.stages, schedule.stages[k], lam, 2.0))


def test_hot_region_of_the_complex_leg(dipeptides_in_water):
    from cufep.config.schema import Rest2Config
    from cufep.workflow import WorkflowError, rest2_region

    system, top, pos, box, ligand, other = dipeptides_in_water
    same = {c.id: c.id for c in top.chains()}          # prepared chain -> chain of the input
    none, names = rest2_region(Rest2Config(scale=2.0, region="ligand"), top, pos, ligand, other)
    assert none == [] and names == []
    # far cutoff: every residue of the other peptide, whole residues
    hot, names = rest2_region(Rest2Config(scale=2.0, region="pocket", pocket_cutoff_nm=5.0), top, pos, ligand, other)
    assert hot == sorted(other) and names == ["ACE1", "ALA2", "NME3"]
    hot, names = rest2_region(Rest2Config(scale=2.0, region="pocket", pocket_cutoff_nm=0.05), top, pos, ligand, other)
    assert hot == []
    atoms = list(top.atoms())
    hot, names = rest2_region(Rest2Config(scale=2.0, region="ligand", residues=["2"]), top, pos, ligand, other, same)
    assert names == ["ALA2"] and {atoms[i].residue.name for i in hot} == {"ALA"} and len(hot) == 10
    chain = atoms[other[0]].residue.chain.id
    assert (
        rest2_region(Rest2Config(scale=2.0, region="ligand", residues=[f"{chain}:2"]), top, pos, ligand, other, same)[0]
        == hot
    )
    with pytest.raises(WorkflowError):
        rest2_region(Rest2Config(scale=2.0, region="ligand", residues=["99"]), top, pos, ligand, other, same)


def test_a_residue_number_of_two_input_chains_needs_its_chain():
    """Review of PR 1 (P2): residues are named as in the input PDB; prepared chains map to the chains of the
    input, and a number that two input chains have does not say which residue is meant."""
    import openmm.app as app

    from cufep.config.schema import Rest2Config
    from cufep.workflow import WorkflowError, rest2_region

    top = app.Topology()
    for chain_id in ("A", "B", "C"):                     # prepared chains: A and B from input chain X, C from Y
        res = top.addResidue("ILE", top.addChain(chain_id), id="960" if chain_id != "B" else "961")
        top.addAtom("CA", app.element.carbon, res)
    pos = np.zeros((3, 3))
    chains = {"A": "X", "B": "X", "C": "Y"}

    def names(*residues):
        return rest2_region(
            Rest2Config(scale=1.5, region="ligand", residues=list(residues)), top, pos, [], [0, 1, 2], chains
        )[1]

    assert names("X:960") == ["ILE960"] and names("Y:960") == ["ILE960"] and names("961") == ["ILE961"]
    assert rest2_region(Rest2Config(scale=1.5, region="ligand", residues=["Y:960"]), top, pos, [], [0, 1, 2], chains)[
        0
    ] == [2]
    with pytest.raises(WorkflowError, match="'960' is a residue of the chains X, Y of the input PDB; give the chain"):
        names("960")


def test_the_validation_compares_the_system_a_run_samples(dipeptides_in_water):
    """Review of PR 1 (P2): `cufep validate` built its System without REST2 and compared states of scale 1 only,
    while runs use REST2 by default; with region: pocket the scaled protein terms were never compared. The
    validation now builds the System of the run (alchemical_leg, without the restraint) and a schedule with the
    REST2 scale of the run along it."""
    from cufep.config.schema import Config
    from cufep.validate import validation_system, verdict
    from cufep.workflow import alchemical_leg

    system, top, pos, box, ligand, other = dipeptides_in_water
    prep = {"topology": top, "positions": pos, "ligand": ligand, "protein": other,
            "input_chains": {c.id: c.id for c in top.chains()}}
    cfg = Config.model_validate({"system": {"protein": "p.pdb", "ligand": "l.sdf"}, "analysis": {"lrc": "none"},
                                 "sampling": {"rest2": {"scale": 1.5, "region": "pocket", "pocket_cutoff_nm": 5.0}}})
    built, schedule = validation_system(cfg, system, prep, box, "complex")
    assert built.rest2_on and built.hot_atoms == sorted(other)
    assert built.xml == alchemical_leg(cfg, system, prep, box, None, True).xml        # the System of the run
    scales = [st.rest2_scale for st in schedule.states]
    assert scales[0] == scales[-1] == 1.0 and max(scales) == pytest.approx(1.5) and any(1.0 < x < 1.5 for x in scales)
    solvent, _ = validation_system(cfg, system, prep, box, "solvent")
    assert solvent.rest2_on and solvent.hot_atoms == []
    # a validation that compared nothing, or not every leg, has not passed (review of PR 1: two skipped legs passed)
    assert verdict({"complex": True, "solvent": True}) is True
    assert verdict({"complex": None, "solvent": None}) is False and verdict({"complex": True, "solvent": None}) is False
    assert verdict({}) is False and verdict({"complex": False, "solvent": True}) is False


@pytest.mark.parametrize("compared_kT,worst", [(-5.0, 85.0), (-80.0, 160.0), (80.0, 0.0), (180.0, 100.0)])
def test_validate_gives_the_differences_of_every_pair_without_a_verdict(compared_kT, worst):
    """Review of 037bdf1 (P2) and of c691e37 (P2): the pairs that validate compared were chosen by a level of 60 kT, and
    a state 80 kT above the sampled one in the reference and 5 kT (or 80 kT) below it in the backend compared was left
    out. The owner, 2026-11-20: no level of cuFEP's own (60 kT, 0.15 kT, 1e-4). Every pair that is finite on both
    backends is compared, and the numbers are given without OK or MISMATCH. Rows of a two-state system evaluated on
    OpenMM's Reference and CPU platforms: 85, 160, 0 and 100 kT over the two pairs."""
    import openmm as mm
    import openmm.unit as unit

    from cufep.validate import between_state_differences

    kT = (unit.MOLAR_GAS_CONSTANT_R * 298.15 * unit.kelvin).value_in_unit(unit.kilojoule_per_mole)

    def rows(platform, offset_kT):
        system = mm.System()
        system.addParticle(1.0)
        force = mm.CustomExternalForce("lam * c")
        force.addGlobalParameter("lam", 0.0)
        force.addGlobalParameter("c", offset_kT * kT)
        force.addParticle(0, [])
        system.addForce(force)
        context = mm.Context(system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName(platform))
        context.setPositions([mm.Vec3(0, 0, 0)])
        out = []
        for lam in (0.0, 1.0):
            context.setParameter("lam", lam)
            out.append(
                context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole) / kT
            )
        return [out]                                      # one configuration, sampled in state 0

    reference, other = rows("Reference", 80.0), rows("CPU", compared_kT)
    found = between_state_differences(reference, other, [0])
    assert found["worst"] == pytest.approx(worst, abs=1e-4)
    assert found["compared"] == 1 and found["one_sided"] == 0 and set(found) == {"worst", "relative", "compared",
                                                                                "one_sided"}
    # a state that is not finite on one backend only is counted and named, not compared; the state a configuration
    # was sampled in is no pair (its difference is zero by definition)
    one = between_state_differences([[0.0, 1.0]], [[0.0, float("nan")]], [0])
    assert one == {"worst": None, "relative": None, "compared": 0, "one_sided": 1}


def test_a_leg_without_a_pair_finite_on_both_backends_is_not_compared():
    """Review of 7b49066 (P2): with every energy of the backend compared not finite, validate printed "max none over 0
    pairs" and counted the leg as compared (18 pairs finite on one side only). A leg is compared when every backend
    has at least one pair of a configuration and another state finite on both sides; how large the differences are
    decides nothing (0 and 80.7 kT alike)."""
    from cufep.validate import between_state_differences, not_compared_reason, verdict

    rng = np.random.default_rng(5)
    reference = rng.normal(size=(9, 3))
    own = [k % 3 for k in range(9)]
    lost = np.full_like(reference, np.nan)
    compared = {"stock mixed": between_state_differences(reference, reference, own),
                "cuPotential": between_state_differences(reference, lost, own)}
    assert compared["cuPotential"]["compared"] == 0 and compared["cuPotential"]["one_sided"] == 18
    reason = not_compared_reason(compared)
    assert (
        reason == "no pair of states has finite energies on both backends for cuPotential (18 pair(s) finite on one "
        "backend only)"
    )
    assert verdict({"complex": None, "solvent": True}) is False
    for offset in (0.0, 80.679):
        shifted = reference + np.where(np.arange(3) == 2, offset, 0.0)
        found = {"cuPotential": between_state_differences(reference, shifted, own)}
        assert not_compared_reason(found) is None and found["cuPotential"]["compared"] == 18
    # a backend whose own states are finite and every other one is not: still no pair
    own_only = np.where(np.arange(3)[None, :] == np.array(own)[:, None], reference, np.nan)
    assert not_compared_reason({"cuPotential": between_state_differences(reference, own_only, own)}) is not None

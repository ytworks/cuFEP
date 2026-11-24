"""The energy change of a relocated water on the two backends, with TIP3P and with OPC: how far is
each from a tight Ewald reference? (plan docs/plans/2026-11-24-opc-water-model.md)

    scripts/in_docker.sh --gpus 0 python spikes/m32_relocated_water_on_the_backends.py --moves 40

The system, the lambda windows and the proposals are those of the GPU test
``test_the_energy_change_of_a_relocated_water_is_that_of_a_tight_ewald_sum``
(``tests/gpu/test_gpu.py``): two alanine dipeptides in a 3.2 nm box of water, one the ligand, and
proposals of water relocation moves made on the processor from one configuration. The acceptance of
a move needs one number of the backend, the change of the potential energy of the simulation when the
water is put at its new pose. For three windows three Contexts give that change for every proposal:

    stock   stock OpenMM, mixed precision, Ewald tolerance 5e-4 (what a leg runs on)
    cupo    the cuPotential plugin
    tight   stock OpenMM, double precision, Ewald tolerance 1e-6 (the reference)

Printed per water model and window: the largest and the root mean square difference of each backend
from the reference, and of the plugin from stock, in kT. No time is measured.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import openmm as mm
import openmm.unit as u

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from conftest import build_dipeptides_in_water  # noqa: E402

from cufep.alchemy.factory import make_alchemical_system  # noqa: E402
from cufep.alchemy.schedule import fixed_schedule  # noqa: E402
from cufep.backends import get_backend  # noqa: E402
from cufep.sampling.relocation_mc import RelocationMover, relocation_spec  # noqa: E402
from cufep.sampling.states import set_state  # noqa: E402

KT = 8.31446261815324e-3 * 300.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--moves", type=int, default=40, help="proposals per window")
    args = parser.parse_args()
    for water in ("amber14/tip3p.xml", "amber14/opc.xml"):
        system, top, pos, box, ligand, _other = build_dipeptides_in_water(box_nm=3.2, water=water)
        cutoff = next(f for f in system.getForces() if isinstance(f, mm.NonbondedForce)).getCutoffDistance()._value
        relocation = relocation_spec(system, top, pos, box, ligand, cutoff, 0.5, 1, 300.0, trial_poses=256,
                                     inner_fraction=0.5, interval_ps=0.4, radius_margin_nm=0.3)
        alch = make_alchemical_system(system, ligand)
        schedule = fixed_schedule(["discharge", "vanish"], {"discharge": 2, "vanish": 3})
        tight = mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(alch.system))
        for force in tight.getForces():
            if isinstance(force, mm.NonbondedForce):
                force.setEwaldErrorTolerance(1e-6)
        for index in (0, 2, 4):
            state = schedule.states[index]
            proposals = RelocationMover(relocation, None, seed=3)
            moves = [proposals.proposal(np.asarray(pos, dtype=float), np.asarray(box, dtype=float), state, bool(i % 2))
                     for i in range(args.moves)]
            moves = [move for move in moves if move is not None]
            changes = {}
            for name, backend, used in (("tight", get_backend("openmm", precision="double"), tight),
                                        ("stock", get_backend("openmm"), alch.system),
                                        ("cupo", get_backend("cupotential"), alch.system)):
                ctx = backend.make_context(
                    backend.prepare_system(used),
                    backend.make_integrator(300 * u.kelvin, 1 / u.picosecond, 4 * u.femtoseconds), 0)
                ctx.setPeriodicBoxVectors(*[mm.Vec3(*r) for r in box])
                set_state(ctx, state)
                mover = RelocationMover(relocation, ctx, seed=3)
                base = mover._potential(pos)
                changes[name] = np.array([mover._potential(move["y"]) - base for move in moves]) / KT
                del ctx
            size = np.abs(changes["tight"])
            print(f"{water} window {index}: {len(moves)} moves, changes of {size.min():.1f} to {size.max():.1f} kT")
            for name, other in (("stock", "tight"), ("cupo", "tight"), ("cupo", "stock")):
                error = changes[name] - changes[other]
                print(f"  {name} against {other}: largest {np.abs(error).max():.3f} kT, "
                      f"root mean square {np.sqrt(np.mean(error ** 2)):.3f}, mean {error.mean():+.3f}, "
                      f"above 0.1 kT in {int((np.abs(error) > 0.1).sum())} of {len(moves)}")


if __name__ == "__main__":
    main()

"""Lossless JSON round trip for OpenMM topologies (PDB files drop non-standard bonds)."""
from __future__ import annotations

import openmm.app as app
import openmm.unit as unit
from openmm.app import element as elem


def topology_to_dict(topology: app.Topology) -> dict:
    chains = []
    for chain in topology.chains():
        residues = []
        for res in chain.residues():
            entry = {"name": res.name, "id": res.id,
                     "atoms": [[a.name, a.element.symbol if a.element is not None else ""] for a in res.atoms()]}
            code = (getattr(res, "insertionCode", "") or "").strip()
            if code:                                   # PDB insertion code: 898 and 898A are two residues
                entry["icode"] = code
            residues.append(entry)
        chains.append({"id": chain.id, "residues": residues})
    bonds = [[b[0].index, b[1].index] for b in topology.bonds()]
    box = topology.getPeriodicBoxVectors()
    box_nm = None if box is None else [[float(v) for v in vec.value_in_unit(unit.nanometer)] for vec in box]
    return {"chains": chains, "bonds": bonds, "box_nm": box_nm}


def topology_from_dict(data: dict) -> app.Topology:
    top = app.Topology()
    atoms = []
    for c in data["chains"]:
        chain = top.addChain(c["id"])
        for r in c["residues"]:
            res = top.addResidue(r["name"], chain, id=r["id"], insertionCode=r.get("icode", ""))
            for name, symbol in r["atoms"]:
                atoms.append(top.addAtom(name, elem.get_by_symbol(symbol) if symbol else None, res))
    for i, j in data["bonds"]:
        top.addBond(atoms[i], atoms[j])
    if data.get("box_nm") is not None:
        import openmm as mm
        top.setPeriodicBoxVectors([mm.Vec3(*row) for row in data["box_nm"]] * unit.nanometer)
    return top

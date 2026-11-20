"""M5: validation of the membrane orientation against OPM.

Set: X-ray structures annotated as transmembrane by both OPM and PDBTM, with the
OPM-oriented coordinates as reference (membrane normal = z, membrane centre = origin).
For every structure this records the tilt and depth error of each orientation method and
of their consensus. (Whether a system gets a membrane is a user setting; nothing here
classifies proteins.)

    spikes/run.sh spikes/m5_orientation_validation.py --n 250 --workers 48
"""
from __future__ import annotations

import argparse
import io
import json
import multiprocessing as mp
import random
import time
import urllib.request
from pathlib import Path

import numpy as np

OUT = Path("spikes/out/orientation_validation")
SEARCH = "https://search.rcsb.org/rcsbsearch/v2/query"


def _term(attribute, operator, value=None, negation=False):
    p = {"attribute": attribute, "operator": operator, "negation": negation}
    if value is not None:
        p["value"] = value
    return {"type": "terminal", "service": "text", "parameters": p}


def search(nodes, rows=3000):
    query = {"query": {"type": "group", "logical_operator": "and", "nodes": nodes}, "return_type": "entry",
             "request_options": {"paginate": {"start": 0, "rows": rows}}}
    req = urllib.request.Request(SEARCH, data=json.dumps(query).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return [x["identifier"].lower() for x in json.load(r).get("result_set", [])]


def membrane_ids(n: int, seed: int = 20261119) -> list[str]:
    nodes = [_term("exptl.method", "exact_match", "X-RAY DIFFRACTION"),
             _term("rcsb_entry_info.polymer_entity_count_protein", "equals", 1),
             _term("rcsb_entry_info.polymer_entity_count_nucleic_acid", "equals", 0),
             _term("rcsb_entry_info.deposited_polymer_monomer_count", "range", {"from": 150, "to": 900}),
             _term("rcsb_entry_info.deposited_polymer_entity_instance_count", "less_or_equal", 2),
             _term("rcsb_entry_info.resolution_combined", "less", 3.0),
             _term("rcsb_polymer_entity_annotation.type", "exact_match", "OPM"),
             _term("rcsb_polymer_entity_annotation.type", "exact_match", "PDBTM")]
    ids = search(nodes)
    random.Random(seed).shuffle(ids)
    return ids[: int(1.4 * n)]


def fetch(url: str, path: Path) -> bool:
    if path.exists() and path.stat().st_size > 1000:
        return True
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            data = r.read()
        if len(data) < 1000:
            return False
        path.write_bytes(data)
        return True
    except Exception:  # noqa: BLE001
        return False


def analyse(task):
    pdb_id, path, use_memembed = task
    import openmm.app as app
    import openmm.unit as u

    from cufep.prep.orient import (
        Orientation,
        OrientationError,
        consensus_orientation,
        memembed_orientation,
        slab_orientation,
        transfer_energy_orientation,
    )
    rec = {"id": pdb_id}
    try:
        text = []
        for l in open(path):
            if l.startswith("ENDMDL"):
                break
            if l.startswith("ATOM") and l[16] in " A" and l[17:20] != "DUM":
                text.append(l)
        pdb = app.PDBFile(io.StringIO("".join(text) + "END\n"))
        pos = np.array(pdb.positions.value_in_unit(u.nanometer))
        t0 = time.perf_counter()
        ref = Orientation("opm", np.array([0.0, 0.0, 1.0]), np.zeros(3), 0.0)
        methods = {}
        for name, fn in (("transfer", transfer_energy_orientation), ("slab", slab_orientation)):
            try:
                methods[name] = fn(pdb.topology, pos)
            except OrientationError:
                pass
        if use_memembed:
            m = memembed_orientation(pdb.topology, pos)
            if m is not None:
                methods["memembed"] = m
        for name, o in methods.items():
            tilt, depth = o.difference(ref)
            rec[f"{name}_tilt"], rec[f"{name}_depth_A"] = tilt, 10 * depth
        try:
            c = consensus_orientation(pdb.topology, pos, use_memembed=use_memembed, log=lambda m: None)
            tilt, depth = c.difference(ref)
            rec.update(consensus=c.method, consensus_tilt=tilt, consensus_depth_A=10 * depth)
        except OrientationError:
            rec["consensus"] = None
        rec["n_residues"] = pdb.topology.getNumResidues()
        rec["seconds"] = time.perf_counter() - t0
    except Exception as exc:  # noqa: BLE001
        rec["error"] = f"{type(exc).__name__}: {str(exc)[:120]}"
    return rec


def summarise(records, out: Path) -> str:
    mem = [r for r in records if "error" not in r]
    lines = [
        "# Membrane orientation: validation against OPM",
        "",
        f"{len(mem)} transmembrane structures (OPM and PDBTM annotated, OPM coordinates as reference); "
        f"{len(records) - len(mem)} files could not be read.",
        "",
        "| method | structures | median tilt (deg) | tilt < 10 deg | tilt < 15 deg | median depth (A) | depth < 3 A |",
        "|---|---|---|---|---|---|---|",
    ]
    for name in ("transfer", "slab", "memembed", "consensus"):
        rows = [r for r in mem if r.get(f"{name}_tilt") is not None]
        if not rows:
            continue
        tilt = np.array([r[f"{name}_tilt"] for r in rows])
        depth = np.array([r[f"{name}_depth_A"] for r in rows])
        lines.append(f"| {name} | {len(rows)} | {np.median(tilt):.1f} | {100 * np.mean(tilt < 10):.0f}% | "
                     f"{100 * np.mean(tilt < 15):.0f}% | {np.median(depth):.1f} | {100 * np.mean(depth < 3):.0f}% |")
    none = sum(1 for r in mem if "consensus" in r and r["consensus"] is None)
    lines += ["", f"No two methods agree (the run stops and asks for an oriented structure): {none} of {len(mem)}."]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--no-memembed", action="store_true")
    args = ap.parse_args()
    (OUT / "membrane").mkdir(parents=True, exist_ok=True)
    ids_file = OUT / "ids.json"
    if ids_file.exists():
        ids = json.loads(ids_file.read_text())
    else:
        ids = {"membrane": membrane_ids(args.n)}
        ids_file.write_text(json.dumps(ids))
    tasks = []
    for pdb_id in ids["membrane"]:
        if len(tasks) >= args.n:
            break
        path = OUT / "membrane" / f"{pdb_id}.pdb"
        if fetch(f"https://opm-assets.storage.googleapis.com/pdb/{pdb_id}.pdb", path):
            tasks.append((pdb_id, str(path), not args.no_memembed))
    print(f"{len(tasks)} structures", flush=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        records = []
        for i, rec in enumerate(pool.imap_unordered(analyse, tasks, chunksize=2)):
            records.append(rec)
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(tasks)} done ({time.time() - t0:.0f} s)", flush=True)
    (OUT / "records_orientation.json").write_text(json.dumps(records, indent=1))
    text = summarise(records, OUT)
    (OUT / "summary_orientation.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

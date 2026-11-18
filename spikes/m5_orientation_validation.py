"""M5: large validation of membrane detection and orientation.

Membrane set: X-ray structures annotated as transmembrane by both OPM and PDBTM, with the
OPM-oriented coordinates as reference (membrane normal = z, membrane centre = origin).
Soluble set: X-ray structures without any membrane annotation.

For every structure this records the classifier's verdict and, for membrane proteins, the
tilt and depth error of each orientation method and of their consensus against OPM.

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


def id_lists(n: int, seed: int = 20261119):
    common = [_term("exptl.method", "exact_match", "X-RAY DIFFRACTION"),
              _term("rcsb_entry_info.polymer_entity_count_protein", "equals", 1),
              _term("rcsb_entry_info.polymer_entity_count_nucleic_acid", "equals", 0),
              _term("rcsb_entry_info.deposited_polymer_monomer_count", "range", {"from": 150, "to": 900}),
              _term("rcsb_entry_info.deposited_polymer_entity_instance_count", "less_or_equal", 2)]
    membrane = search(common + [_term("rcsb_entry_info.resolution_combined", "less", 3.0),
                                _term("rcsb_polymer_entity_annotation.type", "exact_match", "OPM"),
                                _term("rcsb_polymer_entity_annotation.type", "exact_match", "PDBTM")])
    soluble = search(common + [_term("rcsb_entry_info.resolution_combined", "less", 1.8)]
                     + [_term("rcsb_polymer_entity_annotation.type", "exact_match", v, negation=True)
                        for v in ("OPM", "PDBTM", "mpstruc", "MemProtMD")], rows=6000)
    rng = random.Random(seed)
    rng.shuffle(membrane); rng.shuffle(soluble)
    return membrane[: int(1.4 * n)], soluble[: int(1.2 * n)]


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
    kind, pdb_id, path, use_memembed = task
    import openmm.app as app
    import openmm.unit as u

    from cufep.prep.orient import (Orientation, OrientationError, consensus_orientation, detect_membrane,
                                   memembed_orientation, transfer_energy_orientation)
    rec = {"id": pdb_id, "kind": kind}
    try:
        lines = [l for l in open(path) if l.startswith("ATOM") and l[16] in " A" and l[17:20] != "DUM"]
        # first model only
        text, seen_end = [], False
        for l in open(path):
            if l.startswith("ENDMDL"):
                break
            if l.startswith("ATOM") and l[16] in " A":
                text.append(l)
        pdb = app.PDBFile(io.StringIO("".join(text) + "END\n"))
        pos = np.array(pdb.positions.value_in_unit(u.nanometer))
        t0 = time.perf_counter()
        v = detect_membrane(pdb.topology, pos)
        rec.update(is_membrane=bool(v.is_membrane), confident=bool(v.confident), score=float(v.score),
                   helices=int(v.n_crossing_helices), strands=int(v.n_crossing_strands), reason=v.reason,
                   n_residues=pdb.topology.getNumResidues())
        if kind == "membrane":
            ref = Orientation("opm", np.array([0.0, 0.0, 1.0]), np.zeros(3), 0.0)
            methods = {"transfer": v.transfer if v.transfer is not None
                       else transfer_energy_orientation(pdb.topology, pos)}
            if v.normal is not None:
                methods["slab"] = Orientation("slab", v.normal, v.centre_nm, v.thickness_nm)
            if use_memembed:
                m = memembed_orientation(pdb.topology, pos)
                if m is not None:
                    methods["memembed"] = m
            for name, o in methods.items():
                tilt, depth = o.difference(ref)
                rec[f"{name}_tilt"], rec[f"{name}_depth_A"] = tilt, 10 * depth
            try:
                c = consensus_orientation(pdb.topology, pos, verdict=v, use_memembed=use_memembed, log=lambda m: None)
                tilt, depth = c.difference(ref)
                rec.update(consensus=c.method, consensus_tilt=tilt, consensus_depth_A=10 * depth)
            except OrientationError:
                rec["consensus"] = None
        rec["seconds"] = time.perf_counter() - t0
    except Exception as exc:  # noqa: BLE001
        rec["error"] = f"{type(exc).__name__}: {str(exc)[:120]}"
    return rec


def summarise(records, out: Path) -> str:
    ok = [r for r in records if "error" not in r]
    mem = [r for r in ok if r["kind"] == "membrane"]
    sol = [r for r in ok if r["kind"] == "soluble"]
    lines = ["# Membrane detection and orientation: validation", "",
             f"{len(mem)} transmembrane structures (OPM and PDBTM annotated, OPM coordinates as reference) and "
             f"{len(sol)} soluble structures; {len(records) - len(ok)} files could not be read.", "",
             "## Classification", "", "| true class | called membrane, confident | called soluble, confident | not confident (run stops) |",
             "|---|---|---|---|"]
    for name, rows in (("membrane", mem), ("soluble", sol)):
        a = sum(1 for r in rows if r["confident"] and r["is_membrane"])
        b = sum(1 for r in rows if r["confident"] and not r["is_membrane"])
        c = sum(1 for r in rows if not r["confident"])
        lines.append(f"| {name} ({len(rows)}) | {a} | {b} | {c} |")
    lines += ["", "## Orientation against OPM (membrane set)", "",
              "| method | structures | median tilt (deg) | tilt < 10 deg | tilt < 15 deg | median depth (A) | depth < 3 A |",
              "|---|---|---|---|---|---|---|"]
    for name in ("transfer", "slab", "memembed", "consensus"):
        rows = [r for r in mem if r.get(f"{name}_tilt") is not None]
        if not rows:
            continue
        tilt = np.array([r[f"{name}_tilt"] for r in rows]); depth = np.array([r[f"{name}_depth_A"] for r in rows])
        lines.append(f"| {name} | {len(rows)} | {np.median(tilt):.1f} | {100 * np.mean(tilt < 10):.0f}% | "
                     f"{100 * np.mean(tilt < 15):.0f}% | {np.median(depth):.1f} | {100 * np.mean(depth < 3):.0f}% |")
    none = sum(1 for r in mem if "consensus" in r and r["consensus"] is None)
    lines += ["", f"No consensus (the run would stop and ask for an oriented structure): {none} of {len(mem)}."]
    text = "\n".join(lines) + "\n"
    (out / "summary.md").write_text(text)
    return text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--no-memembed", action="store_true")
    ap.add_argument("--holdout", action="store_true",
                    help="use the structures of the id lists that the main run did not use")
    args = ap.parse_args()
    (OUT / "membrane").mkdir(parents=True, exist_ok=True); (OUT / "soluble").mkdir(parents=True, exist_ok=True)
    ids_file = OUT / "ids.json"
    if ids_file.exists():
        ids = json.loads(ids_file.read_text())
    else:
        mem, sol = id_lists(args.n)
        ids = {"membrane": mem, "soluble": sol}
        ids_file.write_text(json.dumps(ids))
    tasks = []
    used_file = OUT / "used_ids.json"
    used = json.loads(used_file.read_text()) if used_file.exists() else {}
    if not used:                       # main run before this option existed: everything on disk was used
        used = {k: sorted(p.stem for p in (OUT / k).glob("*.pdb")) for k in ("membrane", "soluble")}
        used_file.write_text(json.dumps(used))
    for kind, url in (("membrane", "https://opm-assets.storage.googleapis.com/pdb/{}.pdb"),
                      ("soluble", "https://files.rcsb.org/download/{}.pdb")):
        got = 0
        for pdb_id in ids[kind]:
            if got >= args.n:
                break
            if args.holdout == (pdb_id in used[kind]):
                continue
            folder = OUT / (f"holdout_{kind}" if args.holdout else kind)
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{pdb_id}.pdb"
            if fetch(url.format(pdb_id), path):
                tasks.append((kind, pdb_id, str(path), not args.no_memembed))
                got += 1
        print(f"{kind}: {got} structures", flush=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        records = []
        for i, rec in enumerate(pool.imap_unordered(analyse, tasks, chunksize=2)):
            records.append(rec)
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(tasks)} done ({time.time() - t0:.0f} s)", flush=True)
    name = "holdout" if args.holdout else "main"
    (OUT / f"records_{name}.json").write_text(json.dumps(records, indent=1))
    text = summarise(records, OUT)
    (OUT / f"summary_{name}.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

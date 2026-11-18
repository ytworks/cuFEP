"""M5: features for the membrane / soluble decision on the validation set (reuses downloaded files)."""
import glob
import io
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

OUT = Path("spikes/out/orientation_validation")


def features(task):
    kind, path = task
    import openmm.app as app
    import openmm.unit as u

    from cufep.prep.orient import _residue_table, _segments, detect_membrane, transfer_energy_orientation
    rec = {"id": Path(path).stem, "kind": kind}
    try:
        text = []
        for l in open(path):
            if l.startswith("ENDMDL"):
                break
            if l.startswith("ATOM") and l[16] in " A":
                text.append(l)
        pdb = app.PDBFile(io.StringIO("".join(text) + "END\n"))
        pos = np.array(pdb.positions.value_in_unit(u.nanometer))
        v = detect_membrane(pdb.topology, pos)
        t = transfer_energy_orientation(pdb.topology, pos)
        ca, hydrophobic, exposed, dssp, chain = _residue_table(pdb.topology, pos)
        z = (ca - t.centre_nm) @ (t.normal / np.linalg.norm(t.normal))
        half = 0.5 * t.thickness_nm
        out = {}
        for code, min_len, name in (("H", 14, "helix"), ("E", 5, "strand")):
            segs = _segments(dssp, chain, code, min_len)
            for frac in (0.5, 0.7, 0.9):
                out[f"{name}_{frac}"] = int(sum(1 for a, b in segs if min(z[a], z[b]) < -frac * half
                                                and max(z[a], z[b]) > frac * half))
            out[f"{name}_total"] = len(segs)
        inside = np.abs(z) < half
        rec.update(score=float(v.score), slab_helices=int(v.n_crossing_helices), slab_strands=int(v.n_crossing_strands),
                   verdict=bool(v.is_membrane), confident=bool(v.confident), e_transfer=float(t.score),
                   thickness=float(t.thickness_nm), n_res=int(len(ca)), frac_inside=float(inside.mean()),
                   frac_E=float(np.mean(dssp == "E")), frac_H=float(np.mean(dssp == "H")), **out)
    except Exception as exc:  # noqa: BLE001
        rec["error"] = str(exc)[:100]
    return rec


if __name__ == "__main__":
    tasks = [(k, p) for k in ("membrane", "soluble") for p in sorted(glob.glob(str(OUT / k / "*.pdb")))]
    with mp.get_context("spawn").Pool(48) as pool:
        recs = pool.map(features, tasks, chunksize=2)
    (OUT / "features.json").write_text(json.dumps(recs))
    ok = [r for r in recs if "error" not in r]
    print(len(ok), "of", len(recs), "analysed")
    for kind in ("membrane", "soluble"):
        rows = [r for r in ok if r["kind"] == kind]
        e = np.array([r["e_transfer"] for r in rows]); s = np.array([r["score"] for r in rows])
        print(f"{kind}: E_transfer percentiles 5/25/50/75/95: {np.round(np.percentile(e, [5, 25, 50, 75, 95]), 1)}; "
              f"slab score: {np.round(np.percentile(s, [5, 25, 50, 75, 95]), 2)}")
        for key in ("helix_0.5", "helix_0.7", "helix_0.9", "strand_0.5", "strand_0.7"):
            x = np.array([r[key] for r in rows])
            print(f"   {key}: zero in {np.mean(x == 0) * 100:.0f}%, >=2 in {np.mean(x >= 2) * 100:.0f}%, >=3 in {np.mean(x >= 3) * 100:.0f}%, >=8 in {np.mean(x >= 8) * 100:.0f}%")

"""Turn the unmodified Schrödinger benchmark files into cuFEP benchmark inputs.

For each set this writes ``ligands.sdf`` (one record per compound, named by compound) and
``ligands.csv`` (experimental dG, FEP+ predicted dG, which SDF record was used and why).

TYK2: every compound has one neutral record; nothing to choose.

A2A: the source SDF holds 29 records for 17 compounds (rotamers "U"/"D", enantiomers
"R"/"S", deprotonated forms). cuFEP 0.1 handles neutral ligands only, and FEP+ reports one
record per compound in its results file. The rule is therefore: take the record that the
FEP+ results file reports, which is neutral for every compound.

Run from the repository root:  python -m cufep.benchmarks.curate
"""
from __future__ import annotations

import csv
import re
from pathlib import Path


def read_sdf(path: Path) -> list[tuple[str, str]]:
    records = [r for r in path.read_text().split("$$$$\n") if r.strip()]
    return [(r.split("\n")[0].strip(), r) for r in records]


def tag(record: str, name: str) -> str | None:
    m = re.search(rf"> <{re.escape(name)}>\n(.*)\n", record)
    return m.group(1).strip() if m else None


def formal_charge(record: str) -> int:
    total = 0
    for line in record.split("\n"):
        if line.startswith("M  CHG"):
            parts = line.split()[3:]
            total += sum(int(x) for x in parts[1::2])
    return total


def read_fepplus(path: Path) -> list[tuple[str, float, float]]:
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    return [(r["Ligand name"].strip(), float(r["Exp. dG (kcal/mol)"]), float(r["Pred. dG (kcal/mol)"])) for r in rows]


def compound_id(record_name: str) -> str:
    """'4b D' -> '4b', '4jS' -> '4j', '4kU' -> '4k', 'ejm_31' -> 'ejm_31'."""
    name = record_name.split()[0]
    m = re.fullmatch(r"(4[a-z])[A-Z]?", name)
    return m.group(1) if m else name


def curate(set_dir: Path, sdf_name: str, results_name: str, rule: str) -> list[dict]:
    records = dict(read_sdf(set_dir / "inputs" / sdf_name))
    rows, used, out_records = [], set(), []
    for name, exp, pred in read_fepplus(set_dir / "inputs" / results_name):
        if name not in records:
            raise SystemExit(f"{set_dir.name}: FEP+ row '{name}' has no SDF record")
        charge = formal_charge(records[name])
        cid = compound_id(name)
        if charge != 0 or cid in used:
            continue                                   # charged form or a second record of a compound
        used.add(cid)
        sdf_exp = float(tag(records[name], "r_exp_dg"))
        if abs(sdf_exp - exp) > 0.01:
            raise SystemExit(f"{set_dir.name}: experimental values disagree for {name}: {sdf_exp} vs {exp}")
        lines = records[name].split("\n")
        lines[0] = cid
        out_records.append("\n".join(lines))
        rows.append({"compound": cid, "sdf_record": name, "formal_charge": charge, "exp_dg_kcal_mol": f"{exp:.3f}",
                     "fepplus_dg_kcal_mol": f"{pred:.3f}", "selection_rule": rule})
    from cufep.files import publish_text, publish_with

    publish_text(set_dir / "ligands.sdf", "".join(r + "$$$$\n" for r in out_records))

    def table(fh) -> None:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    publish_with(set_dir / "ligands.csv", table, newline="")
    return rows


def evenly_spaced_subset(rows: list[dict], n: int = 5) -> list[str]:
    ranked = sorted(rows, key=lambda r: float(r["exp_dg_kcal_mol"]))
    idx = [int(round(i * (len(ranked) - 1) / (n - 1))) for i in range(n)]
    return [ranked[i]["compound"] for i in idx]


def main(root: Path = Path(".")) -> None:
    bench = root / "benchmarks"
    tyk2 = curate(bench / "tyk2", "tyk2_ligands.sdf", "tyk2_out.csv", "only record of the compound (neutral)")
    a2a = curate(bench / "a2a", "a2a_hip278_ligands.sdf", "a2a_hip278_sbpkacorr_out.csv",
                 "neutral record reported in the FEP+ results file")
    for name, rows in (("tyk2", tyk2), ("a2a", a2a)):
        exp = [float(r["exp_dg_kcal_mol"]) for r in rows]
        print(f"{name}: {len(rows)} compounds, exp dG {min(exp):.2f} .. {max(exp):.2f} kcal/mol, "
              f"subset by rank: {evenly_spaced_subset(rows)}")


if __name__ == "__main__":
    main()

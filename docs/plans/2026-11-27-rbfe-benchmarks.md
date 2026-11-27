# RBFE ベンチマーク(TYK2・A2A)と、実験値・FEP+・OpenFE との比較表
Status: In progress (branch `feat/rbfe-benchmark-tables`)
Refs: `docs/rbfe.md`("Benchmarks"); `benchmarks/README.md`; IndustryBenchmarks2024 v1.0.0(Baumann et al., doi:10.5281/zenodo.17245550); Ross et al. Commun Chem 2023

## Context
- オーナーの依頼(2026-11-27): RBFE で TYK2 と A2A のベンチマークを流せるようにする。LOMAP などで経路を適切に選ぶ。
  ベンチマークを流し、実験値・FEP+・OpenFE との比較表を作る。
- 2026-11-26〜28 の開発(PR #19)はオーナーの指示で捨てた。このブランチは main(e16c1f1)から始め、そのブランチは使わない。
- オーナーの決定(2026-11-27):
  1. プラグインの積分器が拘束できない水素(1 つの対応原子に 4〜5 本、または長さの違う X–H 拘束)を持つ hybrid の脚は、
     main のまま標準 OpenMM で計算する(表に脚ごとのバックエンドを出す)。
  2. A2A は中性 17 化合物(ABFE ベンチマークと同じ。公開 map の 29 状態は扱わない)。
  3. OpenFE は TYK2 だけ公開値がある。A2A の OpenFE 列は空欄で、公開値がないと書く。

## Approach
- 経路: `cufep rbfe` の LOMAP(lomap2 の既定、Kartograf、score 0.4、全化合物を cycle に、経路長 6 以下)。TYK2 16 化合物で
  22 辺。OpenFE の公開ベンチマークの 22 辺と同じで、向きも同じ(テストで固定)。公開 FEP+ map(24 辺)と共通なのは 11 辺。
  A2A 17 化合物で 21 辺。公開 map のうちこの 17 化合物(`ligands.csv` の記録)どうしの 19 辺と共通なのは 10 辺。
- `cufep benchmark run|plan|report NAME --protocol rbfe`(`src/cufep/benchmarks/relative.py`): 設定は RBFE example から取り、
  入力は set の protein と `ligands.sdf`、実験値は `ligands.csv`。tier は smoke(example の最初の辺、短縮)と full。
- 比較表(`rbfe_report.md`、`rbfe_report_ligands.csv`、`rbfe_report_edges.csv`、図 2 枚): 化合物の dG と辺の ddG を、
  cuFEP・FEP+・OpenFE それぞれ同じ化合物・同じ辺の上で実験と比べる(95 % bootstrap)。dG の統計と図は、どの手法も比べる
  化合物の上で実験値の平均に中心をそろえる。
- OpenFE の値: IndustryBenchmarks2024 v1.0.0 の `combined_pymbar3_*` から TYK2 の行をそのまま(`benchmarks/tyk2/inputs/`)。

## Invariants
- 計算の中身は変えない(main の `cufep rbfe` のまま)。判定は文献の基準だけ(NOT RELIABLE は重なり 0.03)。表は数字だけで判定を書かない。
- 1 辺 1 計算(`sampling.repeats: 1`)。FEP+ の名は `scripts/check_names.py` が許すファイルだけ。プラグインの時間は書かない。

## Verification
- CPU テスト `tests/unit/test_benchmarks_rbfe.py`: 設定(example の設定・set の入力・全化合物・LOMAP)、smoke、CLI、公開 map の
  向き、TYK2 の LOMAP 網が OpenFE の 22 辺と同じこと、既知の値での比較表(向きの反転、ネットワーク差、ddG のない辺、
  標準 OpenMM の脚、統計)。
- 実行(コミットから作ったイメージ、8 GPU): TYK2 smoke → TYK2 full → A2A full。

## Record
- 2026-11-27 TYK2 smoke(image cufep:39ee4ac、`runs/rbfe_bench_tyk2_smoke_39ee4ac`): exit 0、ejm_31→ejm_46 ddG
  −1.21 ± 0.88 kcal/mol(実験 −1.77。0.1 ns/窓の短縮)、2 脚とも cuPotential、窓 9 と 9。
- 2026-11-27 TYK2 full を image cufep:39ee4ac で開始(`runs/rbfe_bench_tyk2_39ee4ac`、8 GPU)。
- 2026-11-27 差分の独立レビュー(3 件、各々を修正前のコードで失敗するテストに): (1) A2A の公開 map は記録名("4e D"、4jS …)で
  書かれ、化合物名で引いていた → `ligands.csv` の `sdf_record` で化合物に直し、他の状態の辺は使わない(19 辺のうち 8 辺が
  抜けていた)。(2) dG のない実行で「全辺」の行が空 → ネットワークの ddG が無ければその列を比べない。(3) dG の統計で cuFEP
  だけが比べる化合物の実験平均に合っていた(A2A の FEP+ は公開値のままで平均誤差 −0.156、MUE 0.841、揃えると 0.793)→
  どの手法も比べる化合物の上で中心をそろえる(cinnabar の図の centralizing)。あわせて OpenFE の ± が 3 反復の標準偏差だと明記。
- 2026-11-27 全体の単体テスト(383758f、cufep:dev): 816 passed。最初の回(39ee4ac)の 1 件の失敗は、報告の 5 ファイルが出力の
  ルートの予約名に無かったこと(383758f で追加)。CI(harness、unit)通過。
- 2026-11-27 TYK2 full(image cufep:39ee4ac、8 GPU、`runs/rbfe_bench_tyk2_39ee4ac`): exit 0、22 辺すべてに ddG、16 化合物
  すべてに dG。報告は 383758f のコードで実行のファイルから作った(`benchmarks/results/tyk2_rbfe/`)。kcal/mol、95 % bootstrap、
  各手法を同じ 16 化合物の上で中心をそろえて: dG の RMSE cuFEP 0.67 [0.34, 1.00]、FEP+ 0.47 [0.29, 0.65]、OpenFE 0.59
  [0.42, 0.74]; MUE 0.48、0.37、0.48; pairwise ddG RMSE 0.98、0.69、0.86; R² 0.73、0.86、0.80; Kendall τ 0.70、0.81、0.68。
  辺の ddG の RMSE: 22 辺で cuFEP 0.93 [0.51, 1.29]、OpenFE(同じ 22 辺)0.61 [0.48, 0.72]; 公開 map と共通の 11 辺で
  cuFEP 1.23 [0.64, 1.73]、FEP+ Bennett 0.58 [0.27, 0.81]。
- 5 辺の 10 脚が標準 OpenMM(ejm_31~ejm_45、ejm_42~ejm_54、ejm_43~ejm_42、ejm_50~ejm_42、ejm_50~ejm_55。CCMA の要る拘束 4 本)。
  NOT RELIABLE は ejm_31~ejm_45 の complex 脚だけ(最小の重なり 0.026)。最も外れた化合物は ejm_54(+2.04): ejm_42→ejm_54
  +1.46(実験 −0.75)、ejm_55→ejm_54 +1.40(実験 −1.32)。窓は 9 か 10 で、最小の重なりは 0.026 から 0.22。
  サイクルの閉じ −1.04 から +0.27(7 個)。
- 2026-11-27 A2A full を image cufep:383758f で開始(`runs/rbfe_bench_a2a_383758f`、8 GPU)。


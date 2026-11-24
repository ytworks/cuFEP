# 水モデル OPC: 文献の調査、ベンチマークでの比較、既定値の判断
Status: In progress (2026-11-24: 計画を承認。GPU テストを OPC に広げ、落ちた 1 本をオーナーの決定で Ewald 1e-6 の基準に対する形に直した(下の「記録」)。unit 707 本、GPU 31 本が通った。TYK2 の run を 1 本ずつ始める。二重層の 4 サイトの水は未実装)
Refs: this pull request; `docs/configuration.md` (`forcefield.water`); `docs/protocol.md` ("Systems"); `benchmarks/results/`

## Context

オーナーの依頼(2026-11-24): ABFE では水モデルを OPC にすると精度が明確に良いという論文がある。それを
調べ、水モデルを変えてベンチマークで精度の変化を確かめ、良ければ既定値を変える。

調べた結果:

- Li & Minkara, J. Chem. Inf. Model. 2024, 64, 9459(オーナーが指した論文。本文は公開版 PMC11684738 で
  全文を読んだ。Supporting Information は取得できていない)。タンパク質–糖鎖の 6 複合体(1I82、1I8A、
  2AAC、2ARC、2GGU、2NN8)、Amber22、BAT.py v2.2 を改変した手順(並進・回転の拘束に加えてタンパク質と
  リガンドの配座の拘束、decoupling は 12 窓、拘束は 16 窓、窓あたり 10 ns、MBAR、3 回ずつ)、
  ff19SB + GLYCAM06j、cutoff 10 Å、0.15 M NaCl(イオンのパラメータは書かれていない)、水モデル 5 種。
  ABFE の MUE / RMSE (kcal/mol、Table 4): TIP3P 3.52 / 4.17、OPC 1.57 / 1.85、SPC/E 3.02 / 3.53、
  TIP3P-FB 3.41 / 3.83、TIP4P-Ewald 2.42 / 3.05。
  - 理由として本文が書くこと: "the OPC water model consistently produces less negative free energy
    values compared to TIP3P"(水中のリガンドの decoupling)、"The more negative decoupling free energies
    in bulk water for the other water models lead to overestimated ABFE values."
  - TIP3P の RMSE について: "This elevated RMSE is largely influenced by a significant outlier observed
    for the 1I82 protein–glycan system."
  - 範囲について: 結論は "specific to protein–glycan systems"、"TIP3P may perform well in other systems,
    such as small molecules without standard force fields"。
  - 本文に無いもの: 平均符号付き誤差、相関、複合体ごとの値(Table S8)、自由エネルギーの成分(Figure S4)。
- 逆の結果: Karrenbrock et al., J. Phys. Chem. Lett. 2024(BRD4、ff19SB + GAFF2)。OPC は結合を強く
  見積もり(MUE 1.5 kcal/mol)、TIP3P で 0.8 になった。
- cuFEP の組(ff14SB + OpenFF 2.2.1、薬らしいリガンド)で OPC と TIP3P を比べた文献は見つからなかった。
  OpenFF 2.2.1 は TIP3P と組んで作られている(Sage の論文)。既定値はベンチマークの結果で決める。

## Approach

- 変えるのは水だけ: `forcefield.water: amber14/opc.xml`。タンパク質は ff14SB のまま(論文の組は
  ff19SB + OPC)。既定値は、下の結果が出てオーナーが決めるまで `amber14/tip3p.xml` のまま。
- 水溶性の系: OPC は今も設定できる(`src/cufep/prep/water.py`)。コードの変更は無い。
- cuPotential と 4 サイトの水: GPU テストの系(`rest2_leg`)を TIP3P と OPC の両方で作る。状態ごとの
  エネルギー、2 つのバックエンドの一致、レプリカ交換のテストが両方の水で走る。
- 二重層: 今は 4 サイトの水を設定の段階で拒否している。OpenMM の `Modeller.addMembrane` は 3 サイトの
  水を持つ脂質のパッチから System を作るので、4 サイトの力場ではテンプレートが合わない。二重層を
  3 サイトの水で組み、最後に全ての水へ追加のサイトを足す(結晶水と同じ方法)。拒否を外す。

## Invariants

- 新しい判定は足さない。結果は数値で出し、cuFEP 自身の判定は書かない。
- 既定値は文献と公開された方法に従う。OPC を既定値にするなら、根拠(上の論文、cuFEP のベンチマーク、
  Lipid21 と OPC の組の文献)を `docs/configuration.md` に引用する。
- 1 リガンド 1 計算。自動の動作は足さない。
- 解析の数値は変えない(`ANALYSIS_VERSION` 10、`tests/unit/data/analysis_golden.json`)。
- cuPotential プラグインの速度や実行時間は書かない。

## Verification

- CPU: `scripts/in_docker.sh python -m pytest -q`。GPU: `scripts/in_docker.sh --gpus 0 python -m pytest tests/gpu -q`。
- Runs: コミットから作ったイメージで、セットの設定(5 ns、1 計算)に `forcefield.water: amber14/opc.xml`
  だけを足し、1 リガンドずつ全 GPU で実行して 1 本ごとに報告する。TYK2 16 本(ejm_31、ejm_49、残り)、
  セットの統計、オーナーの判断の後に A2A 17 本(4k、4e、残り)。
- 比較対象: `benchmarks/results/` の全体の run(コミット e77a6c5、TIP3P)。その run で複合体レグの窓の
  始め方が違ったリガンドは、歩行で始めた TIP3P の値があればそれと比べ、無ければ表に印を付ける。
- 出すもの: リガンドごとの結合自由エネルギーと標準誤差、実験値、TIP3P の値との差、溶媒レグと
  複合体レグそれぞれの差、拘束、窓の数、overlap、独立なサンプルの数、レグが走ったバックエンド。
  セットでは、平均符号付き誤差、平均を除いた RMSE、Kendall tau(95 % ブートストラップ区間)を TIP3P と
  並べる。A2A では二重層の数値も並べる。1 計算のばらつき(標準偏差 1.6〜1.7 kcal/mol、ejm_49 と 4k)
  からセットの平均誤差は 0.4 kcal/mol ほど揺れるので、それより小さい差は差として書かない。

## 記録

### 2026-11-24: GPU テストを OPC に広げた。1 本が落ち、run は始めていない

- `rest2_leg`(2 つのジペプチド、3.2 nm の水の箱、REST2)を TIP3P と OPC で作るようにした
  (`tests/conftest.py`、`tests/gpu/test_gpu.py`)。イメージ `cufep:dev`、GPU 1 枚、REST2 のテスト 11 本:
  10 本が通り、`test_rest2_backends_agree[opc]` が落ちた。プラグインと stock OpenMM の、窓のあいだの
  エネルギー差の違いが最大 0.114 kT(テストの水準は 0.1 kT。TIP3P の同じテストは 0.013 kT)。
  OPC での他の 4 本(状態ごとのエネルギー、レプリカ交換、両方のバックエンド)は通った。
- 切り分け(`spikes/m31_water_models_on_the_backends.py`、テストの配置とそれに続く MD の 30 配置、
  8 窓)。Ewald の許容誤差 1e-6・倍精度の stock OpenMM を基準にした、窓のあいだの差の誤差 (kT):

  | 水 | stock (5e-4) 最大 / 平均 | プラグイン 最大 / 平均 | プラグイン対 stock 最大(0.1 を超えた配置) |
  |---|---|---|---|
  | TIP3P | 0.118 / 0.028 | 0.022 / 0.007 | 0.126 (31 のうち 1) |
  | OPC | 0.156 / 0.044 | 0.044 / 0.015 | 0.152 (31 のうち 5) |

  隣り合う窓のあいだでは、プラグインは最大 0.011 (TIP3P) と 0.021 (OPC)、stock は 0.058 と 0.076。
  力の相対差(基準に対して): プラグイン 3.2e-4 (TIP3P) と 4.6e-4 (OPC)、stock 6.8e-4 と 1.09e-3。
  仮想サイトの上の力は 2 つのバックエンドで同じ値。MD は GPU の上なので、テストの配置のほかは実行ごとに
  変わる。
- 読み取れること: プラグインは 4 サイトの水でも基準に stock より近い。0.1 kT を超える分は、許容誤差
  5e-4 の stock の PME の誤差で、TIP3P でも配置によって超える。OPC では両方のバックエンドの誤差が
  TIP3P の 1.3〜2 倍になる。
- 計画の「一致しなければ止めて報告」に従い、TYK2 の run は始めていない。テストの水準は変えていない。

### 2026-11-24: テストを Ewald 1e-6 の基準に対する形に直した(オーナーの決定)

- `test_rest2_backends_agree` は、それぞれのバックエンドの窓のあいだの差を、Ewald の許容誤差 1e-6・
  倍精度の stock OpenMM に対して保つ。水準はプラグイン 0.1 kT、stock(許容誤差 5e-4)0.3 kT で、上の表の
  最大値(0.044 と 0.156)のおよそ 2 倍。テストの配置での値: TIP3P でプラグイン 0.006、stock 0.009、
  OPC でプラグイン 0.009、stock 0.110 kT。
- 同じ形(プラグイン対 stock を 1 配置で 0.1 kT)の TIP3P のテスト 3 本
  (`test_backends_agree_at_every_state`、`test_backends_agree_with_annihilated_electrostatics_and_the_squared_soft_core`、
  `test_the_system_of_a_validation_with_a_rest2_pocket_agrees_on_both_backends`)は変えていない。
- unit テスト 707 本、GPU テスト 31 本が通った(イメージ `cufep:dev`、GPU 1 枚)。`scripts/lint.sh` も通った。

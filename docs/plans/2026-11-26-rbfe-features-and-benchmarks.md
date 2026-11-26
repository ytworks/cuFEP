# RBFE: 公開プロトコルにあって cuFEP に無い機能の実装と、TYK2・A2A フルベンチマークの RBFE 化
Status: In progress
Refs: `docs/plans/2026-11-25-rbfe.md`; `docs/rbfe.md`; Ross et al. Commun. Chem. 2023, 6, 222 (本文と SI);
Chen et al. J. Chem. Theory Comput. 2018, 14, 6346; de Oliveira et al. J. Chem. Theory Comput. 2019, 15, 424;
Liu et al. J. Comput. Aided Mol. Des. 2013 (LOMAP)

## Context
- オーナーの依頼(2026-11-26): FEP+ の仕組みを調べ、cuFEP の RBFE で既定でない・未実装の機能を実装する。TYK2 と A2A の
  フルベンチマークを RBFE で流せるようにし、経路は LOMAP などで精度を落とさずに絞る。
- 読んだ一次資料: Ross et al. 2023 の本文(Methods)と SI(FEP+ 21-3/21-4 によるベンチマークの手順、A2A の準備)、
  FEP+ 2022-1 の `fep_plus` のコマンドと msj 生成のソース(Python API のソース)、FEP+ の hot atoms のヘルプ。
- オーナーの回答: GCMC は RBFE の complex leg で既定 on(ABFE は off のまま)。A2A は公開予測と同じ 29 状態と補正で流す。

## 公開プロトコルの仕組みと扱い
| 公開プロトコル(出典) | cuFEP RBFE の今 | 扱い |
|---|---|---|
| complex leg は GCMC(μVT)が既定、溶媒は NPT(`-ensemble muVT`;Ross 2023 "All complex leg simulations were run in the μVT ensemble") | ABFE だけ(既定 off)、RBFE は拒否 | RBFE に実装、complex leg で既定 on |
| 電荷が変わる摂動: Na+/Cl- と水を同時に変換して箱の電荷を保つ。中和イオンと 0.15 M NaCl はこの摂動だけ。24 窓(Ross 2023;Chen et al. 2018) | 書く前に拒否 | 実装。Chen 2018 を通読し規則を `docs/rbfe.md` に引用。窓は ALS が決める |
| プロトン化・互変異性の状態を map に入れ pKa で補正(de Oliveira et al. 2019) | なし | 実装: 状態ごとの溶液 pKa と pH を YAML で与える |
| 回転異性体・結合様式を map に入れ SI 式 2 で合成。complex で回らない対称な R 基は kT ln 2 下げる(Ross 2023 SI 1.2、1.2.1) | 1 化合物 1 記録 | 実装: 状態を化合物にまとめる。対称性補正は化合物ごとに YAML で |
| cycle closure correction の各エッジの ddG。pairwise RMSE・edgewise RMSE・R²・τ で評価(Ross 2023) | MLE の dG と cycle closure | 追加: ネットワークから出る各エッジの ddG と、これらの指標 |
| custom core(SMARTS、`-atom-mapping`、選択) | なし | 実装 `mapping.core_smarts`(既定なし) |
| REST の hot region の既定は変わる原子の断片(重原子 24、環を含めば 30 まで)、All と protein residue は選択 | リガンド全体 + pocket(オーナーの選択 2026-11-20) | `sampling.rest2.region: perturbed` を追加、既定は pocket のまま |
| 窓は固定で 12、core hopping・大環状化 16、電荷 24 | adaptive lambda scheduling(9 から) | 変えない(主機能) |
| 交換 1.2 ps、5 ns/窓(`-time 5000`)、HMR | 2.5 ps(OpenFE)、5 ns、HMR | 変えない |
| LOMAP 系の map、全化合物が cycle に | LOMAP | そのまま使う(下) |
| core hopping・大環状化(環の切断、Wang et al. 2017) | なし | 実装しない: TYK2・A2A の経路に無い。別の計画で論文を読んでから |
| ねじれの QM 当てはめ(選択) | なし | 実装しない: 既定でなく、イメージに QM が無い |
| タンパク質変異・選択性・溶解度・金属・共有結合の FEP | 対象外 | 対象外(リガンドの RBFE ではない) |

## 経路を絞る
- 規則: LOMAP(lomap2 の既定、Kartograf、score 0.4、全状態が cycle に入る、経路長 6 以下)。今のコードで TYK2 16 化合物
  22 本(公開 map 24 本)、A2A 17 化合物 21 本。29 状態の本数は手順 5 で数える(公開 map 46 本)。
- 精度の確認(GPU の前、CPU だけ、`spikes/m39`): 公開されているエッジの結果(Bennett ddG と SE)で、(a) 公開 map の全
  エッジ、(b) その中から同じ LOMAP の規則で選んだエッジ、から dG を出し、実験値に対する pairwise RMSE・R²・τ
  (bootstrap 95 %)と MLE の予測 SE を並べる。cuFEP の map と公開 map の予測 SE(エッジの SE を同じとして)も。
  数字を報告し、(b) が (a) より悪ければ止めて相談する。

## ベンチマークの RBFE 化
- `cufep benchmark run tyk2|a2a smoke|subset|full --protocol rbfe`、`report`、`freeze`。registry に RBFE の example と
  状態のファイル。A2A の 29 状態は `benchmarks/a2a/rbfe_states.csv`(記録 → 化合物、電荷、対称性補正 10 化合物、溶液
  pKa 4e 8.43・4f 10.72・4o 9.84 と pH 7.5、SI 表 16・17)を `curate.py` が未改変の入力から作る。
- レポート: 化合物の dG を実験・FEP+・cuFEP ABFE(e77a6c5)と、エッジの ddG を実験・FEP+(同じエッジ)と比べ、
  pairwise・edgewise RMSE、RMSE、MAE、R²、τ(bootstrap 95 %)。FEP+ の公開値は 20 ns/窓(Ross 2023)と注記。

## 手順(ブランチ `feat/rbfe-benchmarks`、draft PR、手順ごとに push と報告)
1. この計画。
2. ベンチマークの RBFE 経路と指標、各エッジのネットワーク ddG。`spikes/m39` の確認を報告。
3. RBFE の GCMC(complex leg、既定 on、球は hybrid の core の中心、電荷変換に使う水は対象外)。
4. 電荷が変わるエッジと電荷を持つリガンド(RBFE だけ)。PROGRESS の Product の「中性のみ」を RBFE について改める。
5. 多状態の化合物(SI 式 2、kT ln 2、de Oliveira 2019 の pKa 補正、各論文を通読して規則を引用)と A2A の状態ファイル。
6. 選択肢 `sampling.rest2.region: perturbed`、`mapping.core_smarts`。
7. `docs/rbfe.md`(根拠は論文で書く)、README 2 言語、PROGRESS。短いレビュー → push。
8. 実行(image はコミットから、8 GPU、1 本ずつ): smoke → TYK2 フル → 報告 → A2A フル(29 状態)→ 報告。
   結果は `benchmarks/results/`、この計画の記録、PROGRESS に。RBFE example の記録は GCMC 前のまま(フルセットに含まれる)。

## Invariants
- 新しい判定は作らない。対称性補正・pKa 補正・結合様式の合成は入力と文献の式から出す数字で、判定ではない。
- 自動の振る舞いは頼まれたものだけ: 状態のまとめ方と対称性補正は YAML で与える(検出しない)。GCMC は設定で、RBFE の
  既定 on はオーナーの決定。
- ABFE の数字、ファイル、既定は変えない(`spikes/m38` の fingerprint で確かめる)。

## Verification
- CPU: 多状態・pKa の式を合成データの真値で。各エッジの ddG が MLE の差。指標を手計算と。電荷変化の hybrid の両端が
  A・B(+ イオン/水)の System と一致。GCMC 付き hybrid の状態エネルギーが各状態の System と一致。書く前の拒否。
  `rbfe run`・`rbfe analyze`・`benchmark run`・`report` で同じディレクトリの状態。各書き込み段階での中断と再開。
  ABFE の fingerprint(`spikes/m38`)と既存テストは不変。
- GPU: 電荷変化と GCMC 付き hybrid の各状態を両バックエンドで(`tests/gpu`)。

## 記録

### 2026-11-26 手順 2: ベンチマークの RBFE 経路、各エッジのネットワーク ddG、経路を絞る確認
- `cufep benchmark run NAME --tier smoke|subset|full --protocol rbfe`(`src/cufep/benchmarks/rbfe.py`): 相対 example の
  設定で、入力・化合物・エッジ(LOMAP)・出力だけ変える。smoke は example の最初のエッジを短く。`cufep benchmark report`
  は相対の出力ディレクトリ(`network.json`)を見分ける。`freeze` は相対の出力を断る(RBFE に `lambda_schedule.file` が無い)。
  レポート: 化合物の dG(実験、FEP+、cuFEP ABFE e77a6c5)、エッジの ddG(自身、ネットワーク、FEP+ の Bennett と CCC)、
  pairwise・edgewise RMSE、R²、τ(bootstrap 95 %)、図 2 枚。
- 各エッジのネットワーク ddG(`network_ddg`、`summary.json` と `edges.csv`): MLE の dG(B) − dG(A) と共分散からの SE。
  脚の失敗したエッジにも出す。
- 読んで分かった誤り: lomap2 の `distance_cutoff` 0.4 は「score 0.6(= 1 − 0.4)未満の組を外す」で、外して残った部品は
  score を問わず最良の組で 2 本ずつつなぐ(`GraphGen`)。`docs/rbfe.md` と設定のコメントは「score 0.4 以上の辺」と
  書いていた。直した。TYK2 の 22 本のうち 8 本が部品をつなぐ辺で score 0.25〜0.37。
- `spikes/m39_network_reduction.py tyk2`(CPU、cufep:dev): 公開されたエッジの Bennett ddG と SE で dG を出し実験値と比較。
  (a) 公開 map の 24 本: pairwise RMSE 0.70 [0.41, 0.93]、R² 0.86 [0.72, 0.96]、τ 0.80 [0.58, 0.96]、RMSE 0.48。
  (b) その中から LOMAP の規則で選んだ 22 本: 0.72 [0.43, 0.95]、0.85 [0.70, 0.96]、0.80 [0.58, 0.96]、0.49。
  エッジの誤差を全部 1 とした dG の平均 SE: (a) 0.690、(b) 0.784、(c) cuFEP が全ペアから選ぶ LOMAP の 22 本 0.773。
  (b) は (a) より pairwise RMSE が 0.02 大きい: 計画どおりオーナーに報告し、手順 8 の前に判断を受ける。A2A は 29 状態の
  補正(手順 5)の後に同じ確認をする。
- テスト(CPU): ネットワーク ddG(木では自身の値と誤差、三角形では閉路の和が 3 等分され SE が s√(2/3)、脚の失敗した
  エッジは経路の和)、相対ベンチマークの設定が example と同じ、レポートの数字が手計算と一致し `benchmark report` で同じ
  文字列、`benchmark run --protocol rbfe` が `run_rbfe` に同じ設定を渡し失敗で 1。

### 2026-11-26 手順 3: RBFE の complex leg の GCMC(既定 on)
- `sampling.gcmc` を RBFE で受け付け、既定 on(`RbfeGcmc`、complex leg だけ。溶媒は NPT)。ABFE の既定は off のまま。
  移動・設定・判定は ABFE のもの(`src/cufep/sampling/gcmc.py`)。設定の照合は共通の `gcmc_problems` に切り出した。
- hybrid のために変えたこと: 水の切り替え(`WaterSwitch`、`ghosts_off`、`every_water_on`)が hybrid のリガンドの LJ の
  力 `cufep_hybrid_lj` の epsilonA・epsilonB にも効く(`lennard_jones_forces`)。試行の代用エネルギーでは、リガンドが両端の
  パラメータで全強度で相互作用する(`stand_in_lambdas`、hybrid の状態には lambda_electrostatics が無い)。その後の全
  ポテンシャルの判定が正すので、標本は変わらない(試行が通る割合だけが変わる)。
- 計画から変えた点: 球の中心は「core の中心」でなく両端のリガンドの重原子全部の中心(A → B と B → A で同じ球。半径も
  同じ規則)。
- テストで見つかった不具合: RBFE で `sampling.gcmc` に一部(例 `attempts`)だけ書くと、既定 on の GCMC が off になった
  (既定を default_factory で入れたため)。RBFE 用の型で `enabled` の既定を true にして直し、回帰テストに入れた。
- テスト: CPU(hybrid の Context で水を切った値が、その水なしで作った hybrid と λ 0、0.3、0.7、1 で 1e-6 kJ/mol 以内で
  一致。作った後に切った hybrid も同じ。hybrid の状態で移動が動き、Context の水の状態とエネルギーが一致。complex leg
  だけに記録される)。GPU(両バックエンドで、切り替えた Context の全状態のエネルギーが全部の水の hybrid と 0.02 kT
  以内、HREX で移動の回数・挿入・セル固定): `tests/gpu` の hybrid と GCMC の 22 件が通過(2 GPU)。

### 2026-11-26 手順 4: 電荷が変わるエッジ(alchemical water)と電荷を持つリガンド(RBFE)
- Chen et al. 2018 は全文が手に入らなかった(Europe PMC に無し、オープンアクセスでない)。規則は全文を読める OpenFE の相対
  プロトコル(`explicit_charge_correction`、perses の `transform_waters_into_ions`、main のソース)のものにし、
  `src/cufep/rbfe/charge.py` と `docs/rbfe.md` に書いた: dq = q(A) − q(B) が ±1 のとき、A の座標で溶質(タンパク質、
  リガンド、脂質)から 0.8 nm 以内に原子の無い水のうち残基番号の最小のものを、B では dq の符号の 1 価イオン(系に最も
  多いもの、無ければ力場の Na+/Cl−)にし、H の電荷を 0 にする。|dq| > 1 は書く前に拒否(OpenFE と同じ)。
- hybrid への載せ方(cuFEP の配置): その水は環境のまま、電荷は NonbondedForce の offset(λ_electrostatics_core、
  プラグインは電荷だけを持つ方針どおり)、LJ は NonbondedForce から外し hybrid の LJ の力に A=水・B=イオンで入れ、
  (水, ほかの環境)の相互作用グループを足した。LJ のカットオフ外の補正は無い(リガンドと同じ)。GCMC の水からは外す
  (残基名 ALW)。両レグに 1 つずつ。ABFE の電荷の拒否はそのまま、RBFE は電荷を持つリガンドを受け付ける。
- 既定 on の GCMC の影響: cuFEP が化学ポテンシャルを持たない水モデル(例 TIP4P-Ew)では、値を与えるか GCMC を切らないと
  RBFE の設定が読み込み時に拒否される(既存の規則)。docs に書き、その水を使う既存のテストでは GCMC を切った。
- 全体テスト(65bb947): 809 passed、1 failed — 相対レポートの出力名が出力ディレクトリの予約名に入っていなかった(手順 2
  の漏れ)。加えた。
- テスト: CPU(水の選び方が規則どおりで毎回同じ、両端の全電荷が等しい、λ 0 と 1 で A と B(イオン)の System に一致、
  NonbondedForce は水の電荷の offset だけ、評価器が各状態の直接計算と一致、|dq| > 1 の拒否、準備した両レグの記録と
  hybrid と GCMC からの除外)。GPU: 電荷変化の hybrid の各状態で両バックエンドが厳しい Ewald 和と一致(hybrid の
  GPU テスト 15 件通過)。

### 2026-11-26 手順 5: 化合物の状態(結合様式、回転異性体、プロトン化の形)と A2A の 29 状態
- 一次資料: Ross et al. 2023 SI 式 2(結合様式、溶液中で入れ替わり complex では入れ替わらない)と 1.2.1(complex で回らない
  対称な基は kT ln 2 下げる)、A2A の節(29 状態、表 16 の対称性補正 10 化合物、表 17 の溶液 pKa、pH 7.5、補正の順)は
  全文を読んだ。pKa 補正の de Oliveira et al. 2019 は全文が手に入らなかった。同じ平衡の式を全文の読める Azimi et al.
  2022(SAMPL8、arXiv:2107.05155)式 10、12〜14 から引用した: dG = −kT ln Σ P0(i) exp(−βdG(i))、P0(A−) = α/(1+α)、
  P0(HA) = 1/(1+α)、α = 10^(pH−pKa)。
- `system.states`(CSV: ligand、compound、form、solution_fraction、symmetry_number)と `compound_dg`:
  dG(化合物) = −kT ln Σ_i P_i exp(−dG_i/kT) − kT ln s。定数は化合物の実験値の平均、状態にも同じ定数。SE は重み付きの
  共分散。`compounds.csv`。状態の一部だけの run はその状態で合成し、そう書く。状態・割合・対称数は入力(検出しない)。
  各化合物が 1 状態のときは今までの `ligand_dg` と同じ(`ligand_dg` をこの関数の上に作り直し、既存のテストは不変で通過)。
- `benchmarks/a2a/rbfe_states.{sdf,csv}` を `curate.py` が未改変の入力から作る(29 記録、名前は空白を `_` に、4e・4f・4o の
  形の割合を pKa 8.43・10.72・9.84 と pH 7.5 から、対称数 2 は表 16 の 10 化合物)。A2A の subset・full はこの状態で、
  smoke は example の 1 エッジを状態なしで流す。レポートは化合物の dG を公開予測と比べ、同じ化合物の状態の間のエッジは
  edgewise の誤差に入れない。
- `spikes/m39 a2a`(公開エッジの Bennett ddG、29 状態、補正込み): (a) 公開 map の 46 本 pairwise RMSE 1.52 [0.83, 1.99]、
  R² 0.53、τ 0.60、RMSE 1.05(Ross 2023 の表 15: 1.58 ± 0.10)。(b) その中から LOMAP の規則で 43 本: 1.52 [0.81, 1.98]、
  0.53、0.62、1.04。等誤差の dG の平均 SE: (a) 0.703、(b) 0.769、(c) cuFEP が全ペアから選ぶ LOMAP の 37 本 0.872。
- テスト(CPU): 化合物の式を合成真値で(結合様式 2、プロトン化の形 0.9/0.1、対称数 2、状態の値、重み、SE を共分散から)、
  状態の設定の拒否(割合の和、対称数の食い違い、名前の無いリガンド、状態の実験値の食い違い)、run と analyze で同じ化合物、
  状態ファイルが入力と表 16・17 どおり、化合物ごとのレポート。

### 2026-11-26 手順 6: custom core を実装、REST2 の perturbed の領域は保留(オーナーの判断待ち)
- `mapping.core_smarts`(既定なし): SMARTS に一致する原子だけを両リガンドで対応させ、残りは各リガンドの unique 原子
  (soft-core で変わる)。エッジの対応と LOMAP のネットワーク(`CoreMapper`)の両方に効く。マッピングの設定を作る 4 か所を
  共通の `mapping_settings` に一本化。新しい設定は既定値のとき、準備した記録とレグの記録に入れない(以前の run を
  そのまま続けられる、`SETTINGS_ADDED`)。`-modify-dihe`(custom core のねじれを途中で 0 にする)は実装しない
  (REST2 が hot region のねじれを s で縮める)。
- `sampling.rest2.region: perturbed`(公開プロトコルの hot region の既定: 変わる原子の断片に回転可能な結合の先を足し、
  重原子 24、環を含めば 30 まで; ヘルプ全文を読んだ)は実装していない。分かった事実: cuFEP の hybrid の REST2 はリガンド
  全体が hot である前提で、3 つの電荷の λ に √s を畳み込んでいる。リガンドの一部だけを hot にするには、hot と cold の
  リガンド原子で別の電荷の λ がプラグインに要る(m36 で確かめたのは global parameter 5 つまで)。計画の想定より大きな
  変更なので、オーナーの判断を待つ(既定は pocket のままなので、今の計算には影響しない)。
- テスト(CPU): custom core の一致・拒否、toluene → phenol で環の炭素 6 個だけ対応、LOMAP がその対応で網を作る、既定値の
  ときに記録が変わらない。

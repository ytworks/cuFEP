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

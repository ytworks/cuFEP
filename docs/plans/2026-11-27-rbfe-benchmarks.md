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
- 2026-11-28 オーナーの質問(閉じた PR #19 の最初の TYK2 記録 08d58d5 との比較)への回答で見つけた不具合と、その扱い。
  比較(同じ計算方法、16 化合物で各々中心合わせ): dG の RMSE 08d58d5 0.59 → 39ee4ac 0.67、pairwise 0.86 → 0.98、R² 0.81 → 0.73、
  τ 0.68 → 0.70、辺の RMSE 0.79 → 0.93。solvent 脚はほぼ 0.2 kcal/mol 以内で一致し、2 SE を超える差は complex 脚の 4 本
  (ejm_47~ejm_50 +1.63、ejm_50~ejm_42 −1.62、ejm_50~ejm_48 −1.51、ejm_55~ejm_54 +1.27)。設定の違いは complex 脚の GCMC
  (08d58d5 は ON、今回 OFF)だけ。コードの違い: main の hybrid は対応原子の proper torsion の REST2 係数を A の項にだけ
  掛けていた(`(1+rest2_ds)*(1-λ)*U1 + λ*U2`)。評価器は群全体を s 倍するので、s ≠ 1 の窓では MBAR に渡すエネルギーと
  サンプリングしたハミルトニアンが違う。PR #19 で直したが(2ecbcd2、08d58d5 は修正後)、main には無かった。
- オーナーの決定「直して影響辺を再計算」: d61c215 で係数を項全体に(トルエン → 4-メチルピリジンの再現テストは修正前に
  失敗: λ 0.25 で 28.12 kJ/mol、s 倍なら 23.72)。B の項を持つ対応原子の proper torsion は、準備済みの辺から数えて TYK2 の
  4 辺(ejm_42~ejm_54 14、ejm_50~ejm_42 2、ejm_50~ejm_55 9、ejm_55~ejm_54 12、各脚)と A2A の 15/21 辺。残り 18 辺の
  TYK2 はエネルギーの式が修正前後で同じ(B の項が 0)なので記録のまま使う。サンプルは System のチェックサムで照合される
  ので、TYK2 は実行のコピー(`runs/rbfe_bench_tyk2_39ee4ac_d61c215`)から 4 辺の脚を除き、`--edge` で 4 辺だけ image
  d61c215 で計算する。A2A(383758f は 42 脚中 6 脚で止めた)は新しいディレクトリ `runs/rbfe_bench_a2a_d61c215` に準備済みの辺を
  写して image d61c215 で最初から(GPU 2〜7)。
- 2026-11-28 オーナーの指示「やはり GCMC を入れて計測したい」: RBFE の complex 脚に GCMC(既定 ON、オーナーの選択)。計測は
  TYK2 と A2A の両方(「今の計算は止めて両方」: d61c215 の TYK2 4 辺の再計算は 8 脚中 4 脚、A2A は 42 脚中 6 脚で止めた)。
  実装: 幽霊の水は NonbondedForce と hybrid 配位子の LJ(epsilonA・B)の両方で切る; 球は hybrid 配位子の重原子(A と B)の
  中心; stand-in は各配位子原子をその窓の λ の電荷と LJ で(ABFE の stand-in のソフトコア); 前平衡化は幽霊を切って
  barostat あり、窓と replica exchange はその平均セルで barostat なし; 結果に GCMC の回数を窓ごとに。設定の検査は ABFE と
  共通の関数(`gcmc_problems`)。YAML が `sampling.gcmc` の一部だけを書くと既定 OFF の型で読まれて切れていた → RBFE 用の型
  (`RbfeGcmcConfig`、既定 ON)。CPU テスト(幽霊が両方の力で無いこと、Context での切り替え、stand-in の端と中間、hybrid での
  移動)と GPU テスト(両 backend での切り替え、GPU の stand-in、hybrid の replica exchange と移動): 23 passed。
- 2026-11-28 A2A(GCMC あり、faa65e0)が 2 回とも約 20 分で CUDA の失敗(プラグインのカーネル、MPS で同じ GPU の全プロセスが
  落ちる)。原因: GCMC の脚はセルを前平衡化の後半の平均セルに固定し、最後の配置の分子を丸ごとそのセルへ移していた。膜の
  平衡化(溶質を保持した 10 ns)でセルがまだ流れていて(4b~4m の z: 五分割の平均 10.37 → 10.73 nm)、平均セルが最後の
  セルより z で 2.3 % 短く、分子が重なった(1 プロセス・1 GPU で再現: 最初の MD でプラグインが失敗、標準 OpenMM は NaN;
  最後のセルからなら正常)。4b~4e 1.3 %、4a~4o 0.9 %、4a~4q 0.05 %。TYK2 は前平衡化に自由 MD が無く平均セル = 最後のセル。
- 修正: GCMC の脚のセルは前平衡化の最後の配置のセル(grand の例と同じ)。ABFE と共通の関数(`alchemical_box`)なので ABFE
  も同じに(ABFE の GCMC は既定 OFF)。`positions_in_cell` は削除。TYK2 の実行(faa65e0)は平均セル = 最後のセルなので
  修正後と同じ計算で、そのまま続ける。A2A は修正後のイメージで、準備済みの辺と終わった前平衡化を写して最初から。
- 2026-11-28 TYK2 full、GCMC あり(image cufep:faa65e0、GPU 0〜2、`runs/rbfe_bench_tyk2_faa65e0`、`benchmarks/results/tyk2_rbfe/`):
  exit 0、22/22 辺、16/16 化合物。セルは平均セル = 最後のセル(自由 MD なし)で cd15025 と同じ計算。complex 脚ごとに受け入れた
  挿入 690〜1121、球の中の水 17〜24(窓の平均)。5 辺の 10 脚が標準 OpenMM(CCMA)。kcal/mol、95 % bootstrap、16 化合物で
  各手法を中心合わせ: dG の RMSE 0.65 [0.46, 0.83](FEP+ 0.47、OpenFE 0.59)、MUE 0.55、pairwise 0.95、R² 0.76、τ 0.60
  (0.81、0.68)。辺の RMSE 22 辺 0.82(OpenFE 0.61)、FEP+ と共通の 11 辺 0.73(FEP+ Bennett 0.58)。
- GCMC の有無(torsion 修正の影響しない 18 辺、準備は同じ、違いは GCMC だけ): 辺の RMSE GCMC なし(39ee4ac)0.58 [0.44, 0.71]、
  あり(faa65e0)0.79 [0.58, 0.98]、PR #19 の 08d58d5(GCMC あり)0.76 [0.51, 0.99]。差 / SE の二乗平均 1.84、2 SE 超え 5 本、
  complex 脚で 1.86。1 辺 1 計算ずつで、同じコードの 2 回でも脚が 0.65〜0.69 動く(PR #19 の記録)。torsion 修正の 4 辺は
  GCMC と修正の両方が変わる: ejm_42→ejm_54 +1.46 → −0.88(実験 −0.75)、ejm_55→ejm_54 +1.40 → +0.26(−1.32)、
  ejm_50→ejm_42 −1.52 → −0.11(−0.80)、ejm_50→ejm_55 −0.61 → −0.88(−0.23)。
- 2026-11-28 A2A full、GCMC あり(image cufep:cd15025、GPU 3〜7 で開始し、TYK2 の終了後に同じイメージでチェックポイントから 8 GPU で
  再開、`runs/rbfe_bench_a2a_cd15025`、`benchmarks/results/a2a_rbfe/`): exit 0、21/21 辺、17/17 化合物、CUDA の失敗なし。complex 脚
  ごとに受け入れた挿入 2700〜4500、球の中の水 5〜9(窓の平均)。4d~4f の両脚が標準 OpenMM(CCMA)。NOT RELIABLE なし(最小の重なり
  0.084)。kcal/mol、95 % bootstrap、17 化合物で中心合わせ: dG の RMSE 1.59 [0.95, 2.14](FEP+ 1.08)、MUE 1.22(0.79)、pairwise 2.33
  (1.58)、R² 0.51(0.50)、τ 0.57(0.59)。辺の RMSE 21 辺 1.57、FEP+ と共通の 10 辺 1.97(FEP+ Bennett 1.42)。外れの大きい化合物:
  4a +3.78、4o +3.38(4o→4e −4.34、実験 −0.72)、4r −2.14、4n −1.71。GCMC なしの A2A(RBFE)は止めたので比べる相手がない。
- 2026-11-28 オーナーのレビュー(review 6094696180、P1 2 件・P2 10 件)。各件を今のコードで再現し、修正前に失敗するテストにして
  から直した(c2f6903):
  - P1 水の化学ポテンシャルの違う辺の統合 → 脚の結果に、サンプルが記録した GCMC の有無・μ'・V° を入れ(`water_moves`)、辺の
    反復間(違えば ddG なし)とネットワークの辺間(違えば dG・サイクルなし)で照合。記録の無い旧結果はサンプルの記録から読む。
    ABFE の反復間の照合は issue #20 のまま(オーナーの以前の指示)。
  - P1 挿入した水の速度 → 水を回した行列で重心まわりの相対速度も回す(`turn_velocities`、RelocationMC と共通の 1 関数)。修正前の
    テスト: 結合方向の相対速度 0.79 nm/ps。**記録した TYK2(faa65e0)と A2A(cd15025)の計算はこの修正の前のもので、流し直して
    いない(ΔG への影響は未測定)。**
  - P2 `auto` の水の値は決めた条件(298.15 K、1 bar、cutoff 0.9 nm)のときだけ。条件外は数値を求めるか GCMC を切るよう、設定を
    読むときに拒否。P2 GCMC なしで作ったディレクトリを既定 ON で再実行 → 脚の前平衡化を、書く前・脚を始める前に照合して拒否し、
    `sampling.gcmc.enabled: false` で続けられると案内。P2 旧規則(平均セル)で走った ABFE の GCMC 脚 → 走ったセル(開始配置、
    無ければ checkpoint)を使い続ける。`positions_in_cell` は戻した(旧規則の脚の開始配置の作り直しと `spikes/m35`)。P2 開始 MD
    の GCMC が GPU の割り当てで変わる → ワーカーと脚のプロセスで同じ関数(`water_movers`、`window_mover_on`)。P2 ΔG の RMSE・MUE
    の bootstrap は標本ごとに中心を合わせ直す。P2 OpenFE の ± は両脚の反復の SD から伝播した値と明記(全 22 辺で一致、
    ejm_49→ejm_50: 0.1364 対 0.0587)。P2 比較表に脚ごとの独立サンプルの最小値と基準未満の脚。P2 LOMAP スコアを summary に入れ、
    `cufep rbfe analyze --write` で実行の表を作り直せる。P2 セットの違うディレクトリの報告は書く前に拒否(ABFE の報告も、全部が
    別セットなら拒否)。P2 スクリプトが cufep から import する名前が実在することのテスト(spikes 53 本・scripts)。
  - 記録の作り直し(image c2f6903、`rbfe analyze --write` と report): ddG と dG は全て同じ。dG の RMSE・MUE の区間が変わった:
    TYK2 RMSE 0.65 [0.44, 0.80]、MUE 0.55 [0.33, 0.72](FEP+ 0.47 [0.28, 0.63]、OpenFE 0.59 [0.38, 0.72]); A2A RMSE 1.59
    [0.80, 2.05]、MUE 1.22 [0.65, 1.78](FEP+ 1.08 [0.58, 1.40])。フラグ: TYK2 は NOT RELIABLE が ejm_31~ejm_48 の complex 脚
    (最小の重なり 0.028。これまでの報告に書き落としていた)、独立サンプル 50 未満が ejm_31~ejm_45(complex 48)、ejm_42~ejm_54
    (solvent 25)、ejm_49~ejm_50(complex 36); A2A は NOT RELIABLE なし、50 未満が 4b~4e(complex 31)、4g~4d(complex 35)。
    水の移動の条件は両セットとも全辺で同じ(μ' −6.09 kcal/mol、V° 0.030345 nm³)。


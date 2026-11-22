# 精度の改善: 誤差の原因の測定と、解決策の実装
Status: In progress (手順 0 済み、手順 1 の 6 run を 2026-11-22 に開始、ほかは未着手)
Refs: this pull request; issue #5; `benchmarks/results/`; `docs/protocol.md` ("Interactions inside the ligand", "Adaptive lambda scheduling", "REST2", "How cuFEP compares"); `docs/configuration.md`

## Context

- 現在の既定値での結果(コミット e77a6c5、1 化合物 1 計算、kcal/mol、`benchmarks/results/`): TYK2 16/16 は平均符号誤差 -1.20、平均誤差を除いた RMSE 1.25、Kendall τ 0.47。A2A 17/17 は -4.14、2.68、0.34。ほとんどの化合物で結合を強く予測している(TYK2 13/16、A2A 15/17)。
- オーナーの依頼(2026-11-22): 誤差の原因の仮説と解決策の候補(issue #5 と、それに続く検討。仮説の記号は下の一覧)を元に、精度を改善する方法を見つけて実装する。
- オーナーの決定(2026-11-22):
  - 同一設定の繰り返しを、いまの既定値のばらつきの測定として行う。既定の 1 化合物 1 計算は変えず、繰り返しを解決策にもしない。
  - REST2 の対照計算を行う。
  - run はできるだけ効率的に、少なくする。
  - 解決策の候補を落とさない(RelocationMC を含む)。
- 方針: 解決策の候補すべてに扱いを決める(下の一覧)。コードの解決策は設定として実装し、既定値はオーナーが決めるまで変えない。保存済みの run と再解析で決まることに新しい run は使わない。新しい機能の評価は、仮説がいちばん強い化合物で 1 回ずつにする。

### 仮説(記号)

| 記号 | 原因の仮説 |
|---|---|
| S1 | リガンドが貫通可能になる区間(sterics 0.4 → 0.2 と見積もり)に窓が足りない |
| S2 | リガンド内部の相互作用を全窓で保つため、配座の入れ替わりが全窓で遅い |
| S3 | REST2 の領域と配置が、遅い遷移と合っていない |
| E1 | 窓の開始構造の作り方が worker の数で変わる |
| E2 | 消去側の窓が定常になる前に production が始まる |
| E3 | 消去端の受容体とポケットの水が、結合構造由来のまま平衡のアポ集団に達していない |
| E4 | A2A は空のポケットの水和と二重層が緩和していない |
| E5 | 位置拘束をかけたままの NPT が箱の緩和を遅らせる |
| O1 | 距離拘束が硬い |
| O2 | complex leg のサンプリング量が少ない |
| O3 | 力場と電荷 |

### 計画を立てるときに確認した事実

1. production の保存フレームに、水と消去側の窓はない。保存されるのは状態 0 のレプリカの蛋白質とリガンドだけ(`src/cufep/sampling/worker.py`、`record and state_index == 0`)。全原子の構造は、各窓の開始(`start_configs.npz`)と終了(`hrex/checkpoint.npz`)の 2 時点だけ。消去側の水と側鎖の時系列は、新しい MD(手順 3)で取る。
2. e77a6c5 の 2 つの実行で、worker 1 個で precalculation を行った complex leg(逐次の経路。`schedule.yaml` の `windows_run_concurrently: false`)は、TYK2 が 4 件(ejm_45、ejm_47、ejm_55、jmc_23)、A2A が 7 件(4c、4h、4j、4k、4n、4q、4r)。
   - 平均符号誤差(`benchmark_report.csv` から): A2A は逐次 7 件が -5.85、並行 10 件が -2.95(差 -2.9 ± 1.1、± は 2 群の平均の差の標準誤差)。TYK2 は -1.42(4 件)と -1.13(12 件)(差 -0.3 ± 0.7)。
   - 化合物が違い、後から見つけた比較なので、原因は決まらない。`PROGRESS.md` の未解決の項目にあった「自由エネルギーはこれに依存しない」は確かめられていない(この pull request で直す)。
3. 経路の違い(`src/cufep/alchemy/schedule.py` の `adaptive_schedule.run_all`): pool がないと、各窓は直前に走った最も近い窓の終わり(complex leg では 0.5 + 0.5 ns のあと)から始まる。pool があると、最初の反復は 1 窓 5 ps の walk で開始構造を作り、全窓を並行で走らせる。あとの反復でも、逐次では `nearest` を窓ごとに取り直し、pool では走らせる前にまとめて取る。
4. 3 つめの作り方がある。`schedule.yaml` は `start_configs.npz` より先に書かれる(`src/cufep/workflow.py`)。その間で中断した leg と `lambda_schedule.file` の leg は、`_walk_schedule`(1 窓 50 ps の連鎖)で開始構造を作るが、`schedule.yaml` の記録は precalculation のままになる。
5. リガンド内部の静電は、全ペアを例外にして電荷積を固定している(`src/cufep/alchemy/factory.py`)。消去する設定はない。`docs/protocol.md` の比較表は、OpenFE が内部の静電を消去することを書いていない。

## Approach

### run の数

| | 無条件 | 条件つき(観測が支持した場合だけ) |
|---|---|---|
| ABFE の計算 | 9 | 最大 6 |
| 既存の run の延長(production を 5 ns から 10 ns へ。準備と precalculation はやり直さない) | 2 | – |
| 通常の MD(GPU 1 枚ずつ、アルケミカル計算なし) | 3 | – |

無条件の 9 回の内訳: いまの既定値の繰り返し 5(ejm_49 を 2 回、4k を 3 回)、REST2 の対照 1、新しい機能の評価 3。full のベンチマークは、結果の報告のあとオーナーの指示で行う。

繰り返しを 2 化合物に絞った代償: 3 回から求めた標準偏差の 95 % 区間は 0.52〜6.3 倍と広い。

### 解決策の候補と扱い

| 仮説 | 解決策 | 扱い | 評価 |
|---|---|---|---|
| S1 | (1-λ)² のソフトコア(Ding et al., arXiv:2603.22274 v3) | **実装する**(4c) | ejm_49 で 1 回 |
| S1 | sterics 0.5〜0.2 に窓を置く、消去端の近くに窓を増やす(設定) | 手順 2 で、その区間の overlap、交換の受理率、前半と後半の差を保存済みの run から出す | 不足が見えた場合だけ 1 回 |
| S1 | 受理率が均一になる窓、熱力学的な距離の窓(コード) | この計画では実装しない。窓を置くのは adaptive lambda scheduling(オーナーの決定、2026-11-21 と 22)。手順 2 で不足が見えたら、その規則の中での対処を提案する | – |
| S2 | 静電をリガンド内部ごと消去する(OpenFE の ABFE、openmmtools) | **実装する**(4b) | ejm_49 で 1 回 |
| S2 | 内部の LJ も消去し、回転可能結合のねじれを下げる(Ding et al. v3) | 4b の評価のあともねじれの遷移が少ない場合に**実装する**(4f)。両 leg の消去端が一致する規則を、論文の該当節の全文で確かめてから | 条件つき 1 回 |
| S2 | リガンドの二面角拘束 | 4b と 4f のあとも配座のコストの取り逃しが残る場合に、別の計画にする。拘束の付加と解除の自由エネルギー項が要る | – |
| S2 | 複数のポーズとプロトン化状態を別々に計算して合成する | しない。配座の遅さとは別の仮説で、対象 pH での存在比が要る | – |
| S3 | REST2 の `scale: 1.0`、`region: ligand`(設定) | `scale: 1.0` を手順 1 で 1 回 | 往復の回数か complex leg の値がばらつきを超えて変わった場合に、次の 1 回 |
| S3 | 交換の間隔を短くする(設定) | 上の条件つきの 1 回の候補 | – |
| S3 | 側鎖のねじれを加速した追加レプリカ(Jiang と Roux 2010) | 新しいコードは書かない。`sampling.rest2.residues` で残基を指定すれば同じ加速になる。手順 3 で遅い側鎖が特定できたら、上の条件つきの 1 回の候補 | – |
| E1 | 窓の開始構造の作り方を 1 つに揃える | **実装する**(4a)。walk と、状態を順にたどる方法(YANK の thermodynamic trailblazing)を設定で選び、worker の数では変わらないようにする | 新しい run なし。4k の 3 回(walk)とベンチマークの値(逐次)を比べる |
| E2 | complex leg の `sampling.equilibration_ns` を長くする(Clark et al. 2023 は 1 窓 3 ns) | 新しい run なし。保存済みのサンプルで、捨てる長さを変えて再解析する | 延長の 2 回で 10 ns まで見る |
| E2 | `walk_ps` を長くする | E1 の「状態を順にたどる方法」で代える | – |
| E2 | アルケミカル計算の前に拘束なしの MD を挟む(`restraints.selection: auto`) | 新しい run はしない。3b45bfe のベンチマークが `auto` だった。ポーズの保持はオーナーの決定(2026-11-21) | – |
| E2 | 各窓の水の占有を開始時に平衡化する(Ben-Shalom et al. 2020) | RelocationMC を precalculation と平衡化から動かすことで代える(4d) | – |
| E2 | 解析で平衡化を自動で捨てる(Chodera 2016、openmmtools) | **実装する**(4e、設定で選ぶ) | 新しい run なし。保存済みのサンプルの再解析 |
| E3 | 消去側の窓を、緩和したアポ由来の構造から始める | 手順 3 で消去端の緩和が production(平衡化と合わせて 6 ns)より長く続いた場合に対照 | 条件つき 1 回(ejm_49 の complex leg) |
| E3 | 蛋白質の配座拘束と解放の補正(Mobley et al. 2007) | E3 が支持された場合に、別の計画にする。追加の leg が要る | – |
| E3 | アポの結晶構造から始める | しない。TYK2 にアポの構造がない | – |
| E4 | RelocationMC(Ding et al. v3) | **実装する**(4d) | 4k で 1 回 |
| E4 | grand canonical Monte Carlo | しない。粒子数が変わり、今の座標の持ち方、交換、MBAR に合わない。粒子数を保つ RelocationMC を先に入れる | – |
| E4 | 入力に結晶水を含める | 手順 2 で、ベンチマークの入力と元の PDB(4GIH、5OLZ)のポケットの結晶水を比べる | 入力に水が欠けている場合だけ 1 回(入力を変えるのでオーナーに諮る) |
| E4 | 二重層の平衡化を長くする(Dickson et al. 2022 は 100 ns) | 手順 3 の 100 ns の MD で、面積が定常になる時間を測る | 傾向がある場合だけ 1 回(4k の complex leg) |
| E5 | 体積と面積の時系列の確認 | 手順 2(保存済みの run) | – |
| O1 | 距離拘束の力の定数を下げる | しない。cuFEP 自身の比較でも Clark et al. 2023 でも差が出ていない | – |
| O2 | production を長くする | 手順 1 の ejm_49 と 4k の 1 回ずつを 10 ns まで延ばす | 延長 2 |
| O3 | AM1-BCC 電荷の配座依存の確認 | 手順 2 で計算する(CPU) | – |
| O3 | 力場の変更 | しない(後回し) | – |

### 手順

| 手順 | 内容 | run |
|---|---|---|
| 0 | ブランチ、この計画、draft pull request。`PROGRESS.md` の PR #8 の状態を直す | – |
| 1 | **すぐ開始する 6 run**(イメージ `cufep:e77a6c5`、ベンチマークの実行が記録した `resolved_config.yaml` と `--ligand`、1 run に GPU 4 枚、2 run を並行)。ejm_49 を 2 回(ベンチマークの 1 回と合わせて 3 回)。4k を 3 回(GPU 4 枚では walk の経路になるので、ベンチマークの逐次の値との比較が E1 の対照を兼ねる)。ejm_49 で `sampling.rest2.scale: 1.0` を 1 回。終わったら ejm_49 と 4k の 1 回ずつを 10 ns まで延ばす | ABFE 6、延長 2 |
| 2 | **保存済みの run の測定と再解析(CPU)。** (a) 経路と誤差、complex leg の段階別の値、解放項、尾の補正、2 つの実行の差の内訳。(b) 33 個の complex leg で、段階ごとの ΔG を production の前半・後半と 5 分割で求める。捨てる長さを変えた場合のセットの統計も出す。(c) 各窓の開始と終了で、リガンドの場所の水(分子の番号つき)と蛋白質の重原子、XY の面積(A2A)を数え、経路別、実行別に比べる。(d) A2A の 17 件の二重層の面積とポケットの水の全時系列(`equil.npz`)。(e) 全リガンドのアミドと尿素のねじれの遷移回数(両 leg)。(f) sterics 0.5〜0.2 の overlap と受理率。(g) 入力と元の PDB の結晶水。(h) AM1-BCC 電荷の配座依存 | – |
| 3 | **消去端の長い MD(GPU 1 枚ずつ)。** ejm_49 と 4k の complex leg を、完全消去の状態(Boresch 拘束あり、REST2 なし)で 50 ns。開始は `start_configs.npz` の最後の窓。10 ps ごとに、リガンドの場所の水(番号つき)と蛋白質の重原子、ポケット残基の χ1、Boresch の 6 座標、箱、全状態の換算ポテンシャルを記録する。4k は、結合状態の二重層の平衡化を 100 ns まで延ばした MD も 1 本 | MD 3 |
| 4 | **実装**(run と並行、下の順) | – |
| 5 | **新しい機能の評価**(機能ごとに、手順 4 のコミットのイメージで 1 回)。ejm_49 で内部の静電の消去、ejm_49 で (1-λ)² のソフトコア、4k で RelocationMC。続けて、一覧の条件を満たした対照だけを行う | ABFE 3、条件つき最大 6 |
| 6 | 結果を表で報告し、既定値の変更を提案して止まる。full のベンチマークはオーナーの指示で行う | – |

- 1 回の評価で読むのは、走行内の観測量(前半と後半の差、水と側鎖の分布、ねじれの遷移回数、往復、overlap)と、手順 1 の同じ化合物のばらつきに対する差。
- スクリプト: 手順 2 は `spikes/m21_stored_run_measurements.py`((b) は製品と同じ推定器、`src/cufep/analysis/mbar.py` の `TimeRangeEstimator.part` と `stage_boundaries`。(e) は `spikes/m16_ligand_torsions.py`)。手順 3 は `spikes/m22_decoupled_end_md.py`。手順 1 の集計は `spikes/m19_window_runs.py`。
- 手順 3 の系は、run が使った系と同じものを作る(`alchemical_leg`、`leg_spec`、保存された条件との一致の確認、`ReplicaWorker`)。最後の状態にいたレプリカは、保存された `states` の最後の行から求める(`checkpoint.npz` の `state_of_replica` は最後の交換のあとの割り当て)。

### 実装(手順 4、この順)

どの機能も、先に出典の該当部分を全文読んで規則を `docs/configuration.md` に引用し、出典が黙っている点は cuFEP 独自と書く。設定として入れ、既定値は変えない。

- **4a、E1: 窓の開始構造の作り方を 1 つにする**(`src/cufep/alchemy/schedule.py`、`src/cufep/workflow.py`、`src/cufep/config/schema.py`)
  - 作り方を設定にする。walk(いまの並行の経路、既定)と、状態を順にたどる方法(いまの逐次の経路)。pool の有無では変わらない: 開始構造を先に決め、そのあと pool で、または 1 つずつ走らせる。あとの反復の `nearest` も、走らせる前にまとめて取る。
  - 作り方を開始構造と一緒に記録する(`start_configs.npz` の中、`schedule.yaml` の provenance)。記録のない古い leg はそのまま続け、結果にそう書く。`schedule.yaml` だけが残った leg の `_walk_schedule` の経路も、記録と合うようにする。
  - precalculation の主 Context を、worker と同じ引数で作る(膜の `box_safety`)。
  - worker を保持し続ける項目(`request_devices(..., pinned=True)`)は速度の問題なので、この計画では触らない。
- **4b、S2: 内部の静電の消去**(出典: openmmtools の `AbsoluteAlchemicalFactory`、OpenFE の `openmm_afe`)
  - 設計の確認で読んだ規則(openmmtools の main f6fe77a、OpenFE の main 9a259eb。実装の前に全文を読み直して引用する): OpenFE は `annihilate_electrostatics=True` と `alchemical_pme_treatment="exact"` を固定で使う。リガンドの電荷は λ に比例するオフセットなので、例外でないペアと周期像は λ² で消える。1-4 の例外の電荷積は λ に比例する。内部の LJ は保つ。
  - `src/cufep/alchemy/factory.py`: 設定が有効なとき、リガンド内部の例外の電荷積を 0 にし、その Coulomb を主グループの `CustomBondForce` に移す。この力は既存の `lambda_electrostatics`(REST2 では λ·√s)を読む。新しいグローバルパラメータは足さないので、`src/cufep/sampling/states.py` の評価器は今の多項式のまま使える。4 結合以上離れたペアは λ²、1-4 のペアは出典どおり λ に比例させる。
  - 設定が無効のとき、System は今と 1 バイトも変わらない。保存された条件には、有効なときだけ記録する(`samples_settings`)。そうしないと、保存済みの leg がすべて拒否される。
  - 扱いの違う leg から結合自由エネルギーを作れないようにする(`CYCLE_CONDITIONS`。記録のない古い結果は「消去しない」と読む)。
  - `docs/protocol.md` の「Interactions inside the ligand」と比較表、`docs/configuration.md` を直す。
- **4c、S1: (1-λ)² のソフトコア**(出典: Ding et al. v3 の式 `r_eff^6 = r^6 + α(1-λ)²σ^6`)
  - `factory.py` のソフトコアの式と `alchemy` の設定に、形の選択を足す。窓は adaptive lambda scheduling が今のまま置く。
- **4d、E4: RelocationMC**(出典: Ding et al., arXiv:2603.22274 v3。公開されているコードは CC BY-NC 4.0 なので、論文の記述から実装する)
  - 最初に 2 つ行う(4a と並行)。論文の該当節と、アルケミカル状態での扱いを全文読む。試行位置のエネルギー評価の費用を spike で測る(GPU 1 枚、アルケミカル計算なし)。
  - 論文は専用のカーネルで多数の試行位置を評価する。OpenMM とプラグインの API で同じ数が出せない場合は、出せる数と受理率を報告して、進め方をオーナーに諮る。
  - 実装: 新しい `src/cufep/sampling/relocation_mc.py` を、各反復の MD のあとに呼ぶ(complex leg、precalculation と平衡化から)。受理は、そのレプリカの状態(λ と REST2)のポテンシャルと、順方向と逆方向の提案確率で決める。REST2 の状態での扱いは論文の範囲外なので、cuFEP 独自と書く。試行と受理の回数を窓ごとに記録し、数値で報告する。
  - 大きい変更なので、別の pull request にする。
- **4e、E2: 平衡化の自動検出**(出典: Chodera 2016、openmmtools の `multistate_analysis`)
  - `src/cufep/analysis/mbar.py` に、捨てる長さを有効サンプル数が最大になる点から選ぶ設定を足す。既定は今の固定 1 ns のまま。
- **4f、S2: 内部の LJ の消去とねじれ**(条件つき、4b の評価のあと)

## Invariants

- 新しい判定は足さない。測定、対照、RelocationMC の受理率はすべて数値で報告する(`.claude/rules/fep-invariants.md`、"The only verdicts")。
- 既定は 1 化合物 1 計算のまま。繰り返しは測定だけに使い、解決策にしない("Specification and defaults")。
- 既定値は、出典を `docs/configuration.md` に書き、オーナーが決めてから変える。REST2 の `scale: 1.0` は対照だけで、既定にはしない。
- 既定の解析の数値は変えない(`ANALYSIS_VERSION` 10 のまま、`tests/unit/data/analysis_golden.json` が通る)。4e は設定で選んだときだけ働く。
- アルケミカル状態と REST2 の状態のエネルギーは、各状態に設定した System のエネルギーと一致する(4b、4c のテスト)。
- cuPotential の速度と計算時間は、追跡されるファイル、コミット、GitHub に書かない。

## Verification

- コード: `scripts/in_docker.sh python -m pytest -q`、`scripts/in_docker.sh --gpus 0 python -m pytest tests/gpu -q`、`scripts/lint.sh`、harness の 3 つのチェック。push の前に、書いていない者による差分全体のレビュー。push のあとに Codex のレビュー。
  - 4a: pool ありとなしで、同じ窓、同じ overlap、同じ開始構造(位置、箱、速度)になる(CPU プラットフォーム、1 スレッド、決定的な力、固定シード。2 回以上の挿入が起きる閾値)。今のコードで失敗することを 1 回確かめる。
  - 4b、4c: 各状態のエネルギーが手で設定した System と一致(fixture `dipeptides_in_water`、REST2 あり・なし)。消去端はリガンドの剛体移動で変わらない。評価器が各状態の設定と一致。設定のない保存済みの条件は既定の run で続けられ、設定を有効にすると拒否される。扱いの違う leg の組み合わせから `dg_bind` が出ない。GPU で 2 つのバックエンドが一致。
  - 4d: 分布が既知の小さい系で、移動がボルツマン分布を保つ(Reference プラットフォーム、固定シード)。中断と再開で同じ結果。GPU で 2 つのバックエンドの受理の判定が一致。
  - 4e: 合成データで、選んだ開始点が既知の値になる。設定しない場合に golden の数値が変わらない。
- 測定と評価: 各 run の `schedule.yaml`(経路、窓)、`result.json`、設定を run のファイルから示す。表はこの計画と `PROGRESS.md` に記録する。
- 長い計算はコミットから作ったイメージで走らせる(`spikes/run_frozen.sh`)。

## 記録

- 2026-11-22: 手順 1 を開始。`runs/acc_base_tyk2_ejm_49_{1,2}`、`runs/acc_rest2off_tyk2_ejm_49`(GPU 0〜3、順に)、`runs/acc_base_a2a_4k_{1,2,3}`(GPU 4〜7、順に)。イメージ `cufep:e77a6c5`。結果はまだない。

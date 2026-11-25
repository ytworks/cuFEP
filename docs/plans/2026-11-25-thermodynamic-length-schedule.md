# λ スケジュール: 熱力学的長さによる配置と重なりの床
Status: In progress (2026-11-25: 段階 0 から 3 を実施。新方式は既定で off。方式はオーナーの指示で改訂した (Approach の冒頭)。段階 4 の検証実行を実施 (各実行に同じイメージの対照)。既定を変えるかはオーナーの判断)
Refs: this pull request; `docs/protocol.md` ("Adaptive lambda scheduling"); `docs/configuration.md` (`lambda_schedule.adaptive`); `docs/plans/2026-11-21-als-initial-window-design.md`

## Context

オーナーの指示 (2026-11-25): λ スケジューリングを、現行の ALS (重なり閾値による逐次挿入、Midgley et al.,
JCIM 2025, 65, 512) から、熱力学的長さによる配置 (A3FE: Clark et al., JCTC 2024, 20, 7806) と重なりの保証を
両立する方式に拡張する。この計画は開発プランで、実装は含まない。既定の挙動は変えず、新方式は opt-in。

要点:

- 新方式は、今の ALS 規則をそのまま走らせ、その窓の上に熱力学的長さの条件で窓を足す (改訂後。Approach の冒頭)。
  窓は取らない、動かさない。最初の案 (全部を等長に置き直す) はオーナーの指示で採らなかった。
- 計量は、事前計算が既に全候補 (101 点) で計算しているエネルギーの中心差分から取る。追加のエネルギー計算も
  ∂H/∂λ の出力も要らない。
- s は「隣接窓の交換受理率の目標」から決める。往復率と実効サンプル数は報告する数にとどめる。
- 最初の段階は GPU を使わず、保存済みの 66 脚で長さと「ガウス予測 対 実測」を測る。既定値はその表を見て決める。

### 現状

| 項目 | 現状 (ファイルと関数) |
|---|---|
| 表現 | `Schedule` (`src/cufep/alchemy/schedule.py:79`)。ステージ→状態は `stage_state` (`src/cufep/sampling/states.py:117`)。REST2 のスケールは脚に沿って λ に連動 (`rest2_scale_along_leg`, :103)。窓は候補の添字で持つ (`src/cufep/alchemy/candidates.py`、101 候補/ステージ) |
| ALS の規則 | `bad_sections` (:215、閾値「以下」と NaN が不足)、`choose_section` (:234、最長→最低重なり→先頭)、`redistribute` (:243、両端固定で1窓足し λ 等間隔)、`als_step` (:261) |
| 反復と停止 | `adaptive_schedule` (:425)。ステージごとに独立、1 反復で各ステージ 1 挿入、未実行の窓だけ走らせる。停止は全ステージが `converged` / `saturated` / `max_windows`。反復数の上限は無い。窓は減らない |
| 重なり | `stage_overlap` (:334、ステージの窓全部の MBAR 行列の超対角、`mbar.overlap_matrix` `src/cufep/analysis/mbar.py:926`)、`pair_overlap` (:350、2 窓だけの行列、0〜0.5)。既定は complex が pair 0.30、solvent が stage 0.18 (`src/cufep/config/schema.py:355`, `AdaptiveConfig` :358) |
| 事前計算 | 交換なしの窓ごとの MD (`run_window` :384、並列は `WindowPool` `src/cufep/sampling/hrex.py:177`)。complex 500+500 ps、solvent 0+50 ps。0.5 ps ごとにそのステージの全候補の換算エネルギーを計算 (`candidate_energies` :323)。配列は (サンプル, 候補)。**メモリ上だけで、保存されない**。残るのは `schedule.yaml` (反復ごとの `superdiagonal` / `pair_overlap`) と `start_configs.npz` |
| 本計算の u | `hrex/samples.h5` と `segment_<n>.h5` に `u` (反復, レプリカ, 窓) を kT で保存、平衡化の反復も残る。`states`、`swaps` (隣接ペアの提案数と受理数) もある。**候補格子のエネルギーは無く、全原子の軌跡も無い** (`checkpoint.npz` は最新のみ) ので、保存サンプルを新しい λ で再評価できない |
| 交換 | 1 窓 1 レプリカ。全ペアからランダムに k³ 回 (上限 5 万) の Metropolis (`mix_replicas` `hrex.py:54`)、2.5 ps/反復。隣接交換ではない |
| ∂H/∂λ | どこにも出していない。λ_elec は NonbondedForce のオフセット (cuPotential が置き換える力)、λ_sterics はソフトコアの CustomNonbondedForce (`src/cufep/alchemy/factory.py:102`)、λ_restraints は線形 |
| 解析と判定 | `analyze_leg` (`mbar.py:1180`)、`quality_checks` (:1001、0.03 と 50)、`count_round_trips` (:1128、回数のみで率は無い)、`swap_acceptance` (:1142)。NOT RELIABLE は `quality_warnings` (`src/cufep/workflow.py:3945`)。`ANALYSIS_VERSION` 10、ゴールデンは `LegAnalysis` の一部と `QUALITY_CHECKS` だけを固定 |
| `cufep plan` | `src/cufep/cli.py:194` → `run.plan` (`src/cufep/run.py:39`)。標準出力だけ (`lambda_window_table` cli.py:84)。ファイルを書かず、run への受け渡しは無い (run が自分で `plan()` を呼ぶ)。適応の窓は予測しない |
| 再開と `--reschedule` | 脚に `schedule.yaml` と `start_configs.npz` があればそれを使い、`lambda_schedule` の設定は適用しない (`stored_windows` `workflow.py:3307`)。`--reschedule` は `archive_leg` (:2259) が `previous_<n>/` へ移し、事前計算からやり直す。事前計算に途中保存は無い |
| テスト | `tests/unit/test_schedule.py` (調和窓 `harmonic_windows` :216、疑似プール :250)、`test_config.py:454` (docs の YAML ブロック = コードの既定)、`test_sampling.py:821` (ゴールデン)、`tests/gpu/test_gpu.py:253` |
| ベンチマークと保存済みの実行 | tyk2 16、a2a 17 (膜)。`runs/benchmark_{tyk2,a2a}_full_e77a6c5` (66 脚)、`runs/tyk2_windows_{a,b,c}_5e2cf33_{1,2,3}`。スパイクは `spikes/m19_window_runs.py`、`m20_replica_exchange_figures.py`、`m9_als_variants.py` (事前計算だけやり直す) |

指示の前提と違う点:

1. **u_kn は「事前計算の中」にだけ揃っている。** 全候補のエネルギーは事前計算のメモリ上にあり、一発の配置はそこで
   行える。保存ファイルからはできない (本計算の u は窓だけ)。
2. **Opt-PSO の出典は Zhang, Giese, Lee, York (JCTC 2024, 20, 3935)。** 窓数を固定し、隣接重なりの分散を最小に
   する方法で、目標値は無い (要約取得のみ、全文は段階 0)。
3. **vanish 端点で σ² の発散は見えない。** cuFEP のソフトコアは Beutler 形 (α 0.5)。保存済み 66 脚では長さの密度は
   λ 0.6 から 0.9 で最大で、端では下がる (「記録」の段階 1)。

既知の実測 (`docs/plans/2026-11-21-als-initial-window-design.md`): complex 脚の往復は 13 実行すべてで 0〜1 回で、
最低隣接受理率が 0.07 でも 0.47 でも変わらない。受理率が最も高い C が ΔG のばらつきも最大だった。

## Approach

**改訂 (2026-11-25、オーナーの指示)。** 下の「手順」の 3 と 4 (開始の窓を残さず等長に置き直す、窓は提案より減ってよい) と、
論点 1 の目的関数 (窓数の最小化) は採らない。オーナー: 今の方法 (開始の窓と重なりの規則) は捨てない、窓は減らさない、
足すかどうかは重なりと長さの両方で決める。確定した方式:

1. 今の規則 (ALS) をそのまま走らせる。
2. その結果の窓の上に、隣どうしの長さが刻み s を超える区間へ ceil(d / s) − 1 窓を足す。窓は取らない、動かさない。
3. 足した窓を走らせ、規則が重なりを読み直し、長さも測り直す。どちらも足さなくなるまで繰り返す。

仕上がりは全ペアが重なりの閾値を超え、かつ長さが刻み以下。`als` の窓は必ず新方式の窓に含まれる。計量 (論点 2)、乖離を数で記録すること (論点 4)、刻みを受理率 0.35 から取ることは変えない。論点 3 は、脚の規則は今の行列のまま
で、長さで足した後の確認だけ pair で読む、に改める。経緯は「記録」の段階 3。

### 手順 (脚ごと、ステージごと。改訂前)

1. **提案。** `initial_windows` の窓を今と同じ事前計算で走らせる (変更なし)。
2. **計量。** 各提案窓で σ = sd[(u(λ+Δλ) − u(λ−Δλ)) / 2Δλ] を自分のサンプルから取る (Δλ = 0.01、端は片側差分)。
   提案窓の上で台形則で積分し、累積長 Λ とステージ長 L を得る (kT)。A3FE 式 19 (g = 1) の量で、A3FE のソースと
   同じ計算 (「記録」の段階 0)。
3. **配置。** s = 2·erfcinv(a\*)、区間数 K = ceil(L/s)。Λ の等分点を、提案窓の間の線形補間で λ に戻し (A3FE と
   同じ)、最も近い候補に置く (両端は固定、s は脚の全ステージで共通)。窓は提案より減ってよい。合計が `max_windows` を
   超えるなら L に比例して K を減らし、注記を残す。A3FE は区間数を round(L/s) で取る。cuFEP は ceil にする
   (独自: 区間の長さが s を超えないようにする)。
4. **実測と修復。** 既存のループに戻る。新しい窓だけ走らせ、`pair_overlap` を測り、脚の `overlap_threshold` 以下の
   区間に `als_step` が 1 窓ずつ足す。停止条件と注記は今のまま。
5. **記録。** `schedule.yaml` の provenance に、方式、a\*、s (kT と kcal/mol)、ステージごとの L と σ、提案の窓、
   配置した窓、ペアごとの予測と実測、修復の反復を書く。

組み合わせ (A3FE の計量、プレプリントの受理率目標、Midgley の規則での修復) は cuFEP 独自として文書に記す。

ガウス近似での換算 (`cufep.alchemy.thermodynamic_length` の関数の値。関数は調和窓のサンプルに対して単体テストで
確かめてある):

| 隣接窓の長さ d (kT) | 交換受理率 erfc(d/2) | pair 重なり | stage 行列の要素 (等間隔) |
|---|---|---|---|
| 0.84 (A3FE の 0.5 kcal/mol) | 0.55 | 0.42 | 0.20 |
| 1.32 | 0.35 | 0.34 | 0.24 |
| 1.55 | 0.27 | 0.30 | 0.24 |
| 1.69 (1.0 kcal/mol) | 0.23 | 0.28 | 0.23 |
| 3.38 (2.0 kcal/mol) | 0.02 | 0.07 | 0.06 |

pair 0.30 は受理率 0.27 に当たる。start A の 3 実行の最低隣接受理率は 0.27〜0.28 だった。

### 論点 1: HREX の下で s を何で決めるか

**決定。** 目的関数は総 GPU 時間 (窓数 × 窓あたりの本計算長) で、これを最小にする。制約は 2 つ: 隣接交換受理率の
目標 a\* (s を決める) と、実測 pair 重なりの床 (修復)。往復率と実効サンプル数は目的関数にせず、報告する。

- A3FE の「できるだけ広く」は持ち込まない。A3FE 自身が 4.2.1 節で、レプリカ交換では交換率の必要から低い速さが
  有利になりうると書いている。2.0 kcal/mol は受理率換算で 0.02。
- 往復率は目的関数にできない。complex 脚では窓間隔を変えても 0〜1 回のままで、交換なしの短い事前計算からは
  推定もできない。
- 実効サンプル数は g で決まり、s の関数ではない。
- cuFEP は窓あたりの長さが設定なので、s を上げると窓と一緒に標本も減る。分散 × コストは一次では s に依らない。
  s は精度とコストの交換比で、上限は制約が決める。
- 交換は全ペアのギブス型で、隣接交換の往復率の式は当てはまらない。
- a\* の仮の既定は 0.35 (Walter & Bereau, arXiv:2610.11714 S1.8: 全 λ のエネルギーから受理率を予測し、ガウス仮定で
  長さに直して等間隔、HREX)。査読前の値 (質問 1)。

### 論点 2: TI が無い中での計量

| 推定法 | 追加コスト | 精度と弱点 | 判断 |
|---|---|---|---|
| A. 提案窓での候補格子の中心差分の分散、窓間は補間 | 無し (エネルギーは計算済み) | 差分誤差は Δλ² の程度。提案が粗いと補間が外れる | **採用** |
| B. 隣接「窓」間の ΔU の分散 (両側から) | 無し | 窓が離れると 2 次の関係から外れる | 照合の数として記録 |
| C. MBAR の再重み付けで全候補の分散 | MBAR 1 回/ステージ | 提案窓の重なりが乏しい所で雑音。cuFEP 独自の拡張 | 不採用。A が足りないと段階 1 で分かれば再検討 |
| D. 勾配の明示出力 | 両バックエンドの力の改修と GPU テスト | 標準の NonbondedForce にパラメータ微分は無い (Custom 系のみ)。プラグイン側は未確認。REST2 が λ に連動し連鎖律が要る。得られる量は A と同じ | 不採用 |
| E. 本計算の u (窓だけ) | 無し | 候補格子が無い。事後の診断にしか使えない | 診断として追加 (段階 2) |

g (統計的非効率) での重み付けは採らない。短い事前計算では g を推定できず (A3FE も式 19 を使う)、g で重み付けた
start B は 3 実行の標準偏差が 1.11 だった (A は 0.17)。A3FE の時間配分 (式 25) も採らない (本計算を自動で延ばさない
規則)。

### 論点 3: stage か pair か

**決定: pair。** 計量は経路の性質で窓の置き方に依らない。床と目標は 2 窓だけで決まる量 (pair 重なり、受理率) で
見る。stage 行列の要素は d について単調でなく (d = √2 で最大 0.24)、d = 0.5 と d = 2.6 を区別できない。新方式の
脚は pair だけを読む。s は脚ごと、配置はステージごと。

### 論点 4: 多峰性の検出

**決定: 予測と実測の乖離を数として記録し、動作は実測値だけで決める。**

- 配置後の各ペアで、ガウス予測 (a\* と対応する pair 重なり) と実測 (2 窓のサンプルからの pair 重なりと受理率) の
  比を記録する。両側の sd(ΔU) の比も記録する。判定は付けない。
- 修復は実測 pair 重なりが閾値以下のときだけ動く (公知の規則)。
- 遅い多峰性 (側鎖の反転、水の出入り) は 0.05〜1 ns の事前計算に現れず、どの指標でも検出できない。本計算の後に、
  窓ごとの統計的非効率、事前計算の予測と本計算の実測の差として数で出す。
- ΔU 分布の多峰性の検定 (dip 検定など) は採らない。有意水準が cuFEP 独自の判定になる。事前計算のサンプルは相関
  していて検定の前提に合わず、サンプルされなかった峰は見えない。

### 論点 5: 局所修復

**決定: 二段構成。** 大域配置の後、実測で床を割ったペアだけ既存の規則で 1 窓ずつ足す。

- 見積もり: K 区間のうち m 区間を修復すると、同じ窓数の等長配置に比べて分散が約 m/(2K) 大きい (K = 30、m = 3 で
  4.5 %)。分散そのものは増えない (修復した区間の寄与は半分になる)。
- 修復件数は実行ごとに記録する。
- 不採用: 床を割ったら全体を置き直す (全窓を走らせ直す)、局所だけ (今の ALS)、窓数固定で隣接重なりの分散を最小化
  (Opt-PSO 型、窓数を人が決める)。

### config

```yaml
lambda_schedule:
  adaptive:
    method: als                  # 新規。als | thermodynamic_length。1 つ、または {complex: ..., solvent: ...}
    thermodynamic_length:        # 新規
      exchange_acceptance: 0.35  # 隣接窓の交換受理率の目標。1 つ、または脚ごと。仮の値 (質問 1)
    overlap_threshold: {complex: 0.3, solvent: 0.18}   # 既存。新方式では修復の床
    overlap_matrix: {complex: pair, solvent: stage}    # 既存。新方式の脚は pair
    # initial_windows (新方式では提案の窓)、candidates、window_ps、equilibration_ps、
    # sample_interval_ps、start_configurations、walk_ps、max_windows は既存のまま
```

- `method` が無い YAML は `als` で、今と同じ窓になる。既存の `resolved_config.yaml`、`schedule.yaml`、
  `lambda_schedule.file` はそのまま読める。
- 新方式の脚で `overlap_matrix` が与えられていなければ pair、`overlap_threshold` も与えられていなければ pair の
  既定 0.30。
- 設定エラー (`configuration_problems`): 新方式の脚に `overlap_matrix: stage` を明示した場合。a\* のガウス換算の
  pair 重なりが脚の `overlap_threshold` 以下の場合 (全ペアが修復になる)。
- 通知: `thermodynamic_length` を書いたが新方式の脚が無い場合。
- `lambda_windows_added` は「規則が足した窓」の意味を保つ (新方式では修復の数)。提案と配置は provenance の新しい
  キーに書く。

### 実装段階

オーナーの指示 (2026-11-25): PR は 1 本。段階はその中のコミットの区切りで、どの段階の終わりでもマージできる状態を保つ。

| 段階 | 内容 | 検証 |
|---|---|---|
| 0 | 出典の全文確認と規則の書き下し (A3FE の SI、Minh、Rizzi、Nguyen & Minh、Zhang et al.、プレプリント)。`PROGRESS.md` に記入。コード無し | `check_progress_size.py`、`check_names.py` |
| 1 | 測定の関数 (新モジュール `src/cufep/alchemy/thermodynamic_length.py`: 勾配の sd、累積長、受理率の予測、ガウス換算)。スパイク `spikes/m36_thermodynamic_length.py` で保存済み 66 脚の本計算 u から、ステージ長、ペアごとの d、ガウス予測対実測、s ごとの予測窓数、事前計算と本計算の pair 重なりの差を出す。挙動は変えない | 単体テスト: 調和窓で長さ = 両端の距離、受理率 = erfc(d/2)、既存の固定値 0.400 / 0.223。既存テスト全通過。GPU 不要。表を報告 |
| 2 | 記録の追加。既定の ALS の事前計算で σ、累積長、最終窓の予測受理率を provenance に書く。本計算の解析にペアごとの長さと予測を数として足す (判定なし) | provenance と解析の新しい数の単体テスト。ゴールデン不変 (`ANALYSIS_VERSION` 据え置き)。`m9_als_variants.py` で保存済みの脚の事前計算だけやり直す (GPU 1 台)。GPU テスト |
| 3 | 新方式 (opt-in)。config、`configuration_problems`、`als_settings`、純関数 `place_by_length`、`adaptive_schedule` の分岐 (提案の後に窓を置き換え、既存ループで実測と修復)、provenance、`lambda_window_table`、文書 | 単体テスト: 等長に置かれ窓数が s どおり、提案より減る場合、床を割るペアだけ修復 (二峰の合成系)、設定エラー、既定 `als` の窓が不変。入口ごとの同一状態 (再開、`analyze`、`benchmark freeze`、`--reschedule`)。GPU テスト (プール)。スモーク階層が終了コード 0。書いていない者の短い差分レビュー |
| 4 | 検証実行 (Verification) と記録。既定は変えない | 結果をこの計画の「記録」、`PROGRESS.md`、`docs/configuration.md` に書く |

## Invariants

- 公知の判定は変えない: 隣接重なり 0.03、窓あたり独立サンプル 50、`windows_determined`、NOT RELIABLE の付け方。
  新しい数 (長さ、予測と実測の比、往復の率) は判定なしで報告する。
- `.claude/rules/fep-invariants.md` の表に新方式の行を足す (段階 3): 計量は A3FE 式 19、目標値はプロジェクトの選択、
  組み合わせは cuFEP 独自。値と出典は `docs/configuration.md` に引用する。
- 解析の数は変えない。`ANALYSIS_VERSION` とゴールデンは据え置き (新しい数は固定された項目の外に足す)。
- 既定は変えない。`auto` を広げない。本計算を自動で延ばさない、やり直さない。
- 1 リガンド 1 計算。V2 の繰り返しはプロトコルの検証のためだけに使う。
- 出典は実装の前に全文を読み、規則を正確に書く (段階 0)。出典が黙っている点は cuFEP 独自と記す。
- cuPotential の速度と実行時間は追跡ファイル、コミットメッセージ、GitHub に書かない。

## Verification

| 実行 | 系 | 比較対象 |
|---|---|---|
| V0 (GPU 無し、段階 1) | 保存済み 66 脚と窓設計の 13 実行 | 予測対実測、s ごとの予測窓数 |
| V1 | TYK2 ejm_31、両脚を新方式、8 GPU で 1 回 | e77a6c5 の ejm_31 (-10.08 ± 0.37、33 / 18 窓、最低重なり 0.154 / 0.189) と start A の 3 実行 |
| V2 (V1 が正常なら) | ejm_31 を同じ YAML と seed であと 2 回 | 3 回の標準偏差を A の 0.17 と比べる |
| V3 | A2A 4e (膜)、8 GPU で 1 回 | e77a6c5 の 4e (-13.50 ± 0.34、30 / 17 窓) |
| 同コスト比較 | 追加実行なし。A の本計算を切り詰めて再解析 (`m19_window_runs.py --production-ns`) | 窓数 × ns を揃えた ΔG と不確かさ |

改訂 (2026-11-25、オーナーの指摘「それをやらずに新方式を実行していることも問題 / 何を試験しているかがわからない」):
比べる相手が別のイメージの実行なら、新方式を走らせる前に、(1) 2 つのコードが既定の経路で同じ System、ワーカーの設定、
解析を作ることをコードと記録で照合し、(2) 同じイメージで今の方式の対照を 1 回走らせる。新方式の実行はその対照と比べる。
V1 と V2 はこの順を踏まずに走らせたので、対照を後から走らせた。OPC の実行と V3 は照合と対照を先にする。

- 1 リガンドずつ 8 GPU で走らせ、終わるごとに報告する。全セットは含めない。イメージはコミットから作る
  (`spikes/run_frozen.sh`)。
- 指標: 窓数 (脚、ステージ)、事前計算で走らせた窓数、修復件数、シミュレーション総量 (窓数 × ns)、隣接重なりの
  最小値と分布 (pair、解析の 2 つの行列)、隣接受理率の最小・最大・変動係数、往復の回数と率 (回/ns)、窓ごとの
  独立サンプルの最少、ΔG ± 標準誤差 (脚別)、実験値との差。
- 壁時計と GPU 時間は `timing.json` から取り、オーナーへの報告だけに書く。

### リスクと早期検出

| リスク | 早期検出 |
|---|---|
| ガウスの関係が広く外れ、修復だらけになる (A3FE は 2.0 kcal/mol で重なり 0.03 付近を観測) | V0: 約 1600 ペアの予測対実測。配置を書く前に分かる |
| 事前計算 (短い、交換なし) の集団が本計算と違い、配置が本計算で外れる | V0: `schedule.yaml` の事前計算 pair 重なりと本計算の値の差 |
| 目標 0.35 (pair 換算 0.34) と床 0.30 の余裕が小さく、雑音で修復が増える | V0: 実測のばらつきから修復の見込みを出す。段階 2 の記録 |
| 実行ごとのばらつきが増える (start B、C で起きた。原因不明) | V2 |
| 事前計算のコスト増 (最終の窓のほぼ全部が新規) | `windows_simulated` と `timing.json` の `lambda_schedule` |
| solvent 脚の σ が雑音 (100 サンプル/窓) で配置が揺れる | 段階 1 の合成テスト。段階 2 で同じ脚の事前計算を 2 回 |
| vanish 端点で σ が大きく、窓が端に集まるか `max_windows` に当たる | V0 で測った: ステージ長は 17 から 22 kT で、端に集まらない (「記録」の段階 1)。実行時は注記 |
| 設定の食い違い (目標が床より下、stage の明示) | 設定エラーとそのテスト |
| 入口による違い (再開、`analyze`、`freeze`、`--reschedule`) | 段階 3 の入口横断テスト |
| 窓が必要以上に近く、解析の stage 行列が 0.03 を割る | 換算表では d ≥ 0.11 (a\* ≤ 0.94) で起きない。V1 の `min_overlap` |

## 質問と未確認

質問 (オーナーの判断)。オーナーの回答 (2026-11-25、V0 の表を見て「上記で ok」): 1 は 0.35、2 は脚の `overlap_threshold`
(pair 0.30)、3 は ejm_31 を 3 回と 4e を 1 回 (1 リガンドずつ 8 GPU)。4 から 7 は計画の案のまま進める (`--reschedule` は今と同じ、
候補エネルギーは保存しない、Opt-PSO は Zhang et al. として扱う、solvent 脚も新方式で検証する)。

1. 目標受理率の既定。0.35 (査読前のプレプリント) を仮に置き、V0 の表を見て決める進め方でよいか。査読前の値を
   根拠にできないなら、A3FE の速さ 0.5〜1.0 kcal/mol (受理率換算 0.55〜0.23) から選ぶ。
2. 修復の床。新方式でも脚の `overlap_threshold` (pair 0.30) を使う案でよいか。
3. 検証実行は ejm_31 を 3 回と 4e を 1 回でよいか。
4. `--reschedule` は今と同じ (事前計算からやり直す) にとどめる。前回の本計算の u を計量の入力に使う拡張は要るか。
5. 事前計算の候補エネルギーは保存せず、導いた曲線だけ `schedule.yaml` に書く案でよいか (生データは complex 脚で
   数十 MB)。
6. 「Opt-PSO (Lee et al.)」は Zhang, Giese, Lee, York の論文を指すか。
7. 検証は solvent 脚も新方式にする案でよいか。

未確認:

- A3FE の SI (S1、S2、S10)、Minh、Rizzi、Nguyen & Minh、Zhang et al. の全文。読んだものは「記録」の段階 0。
- cuPotential のパラメータ微分への対応 (計画は依存しない)。
- 事前計算の勾配から取る長さ (段階 2 で記録する)。「記録」の段階 1 の長さは本計算の隣接窓の距離の和。

## 記録

### 段階 0: 出典 (2026-11-25)

- **A3FE の論文** (Clark et al., JCTC 2024, 20, 7806)。本文の 2.1、2.1.1、3.3、4.1.2、4.2.1 節を読んだ。式は Europe PMC の
  XML では画像で、規則は文面とソースから取った。
  - 2.1.1: "The windows can then be spaced according to eqs 19 or 18 by ensuring that the areas under the [...] curves
    between each value of λ are equal. [...] We specify a constant thermodynamic speed".
  - 4.1.2: "we select λ values to achieve minimum variance using eq 19 based on a short set of initial simulations of
    negligible cost. [...] a single replicate run with 0.1 ns sampling per window. This is effectively equivalent to
    selecting states to achieve equal and sufficient overlap when statistical inefficiency is ignored."
  - 4.2.1: "A speed of 2.0 kcal mol–1 appeared close to the maximum, because low off-diagonal overlaps were observed
    (according to the rule-of-thumb that off-diagonal values should not be lower than 0.03)." "If this method were used
    with replica exchange, benefits would still be expected at relatively high thermodynamic speeds, but the need for a
    sufficiently high exchange rate may favor lower speeds." "This protocol requires that the λ-schedule for the trial
    simulations is sufficiently dense to faithfully capture variations in [...] with λ."
- **A3FE のソース** (github.com/michellab/a3fe、main、2026-11-25 に取得、GPL-3。規則を読むだけでコードは写さない)、
  `a3fe/analyse/process_grads.py`:
  1. `get_integrated_error(er_type="root_var")`: 各 λ の勾配の分散の平方根を、走らせた λ の上で台形則で積分する。
  2. `calculate_optimal_lam_vals`: 値の数は round(全長 / delta_er) + 1、最少 2。0 から全長までを等分し、各点を
     積分値から λ への線形補間で λ に戻し、小数 3 桁に丸める。
  3. `Stage.get_optimal_lam_vals` の説明: "A sensible default [...] for root_var is 2 kcal mol-1"。
  4. `get_predicted_overlap_mat`: 隣接窓の重なりの予測は 1 / (2 + β²Δλ²·(両窓の分散の平均) / 2)。
- **cuFEP が変える点**: 勾配は候補格子の中心差分 (Δλ 0.01)。単位は kT。区間数は ceil。λ は候補に丸める。
- **Walter & Bereau** (arXiv:2610.11714 v1) の S1.8 節を全文読んだ: "During the proposal simulations we computed the
  energy of every sampled configuration in every λ state, so the exchange acceptance can be predicted for any pair of
  states. We used this acceptance as the measure of overlap, since it directly limits the replica exchange and requires
  no additional simulations." "we assumed Gaussian distributed exchange energies. The acceptance of a pair then depends
  only on the distance between the two states, analogous to a thermodynamic length. Summing these distances gives a
  cumulative length along the leg, and equally spaced windows in this length yield a uniform acceptance."
  "Targeting an acceptance of 0.35". 受理率と距離の式は本文に無い。erfc(d/2) は cuFEP が導いた
  (`src/cufep/alchemy/thermodynamic_length.py` の説明、単体テスト)。全リガンド共通のスケジュールを 1 回決める方法で、
  リガンドごとではない。
- **Zhang, Giese, Lee, York** (JCTC 2024, 20, 3935): PMC のページの要約取得のみ。採らない方法なので全文は読んでいない。
- 未読: A3FE の SI (S1、S2、S10)、Minh、Rizzi、Nguyen & Minh。

### 段階 1: 測定の関数と、保存済みの実行の測定 V0 (2026-11-25)

実装: `src/cufep/alchemy/thermodynamic_length.py` (勾配の標準偏差、台形則の累積長、隣接窓のエネルギー差、距離、
サンプルの全ペアでの交換受理率、ガウスの換算)、`tests/unit/test_thermodynamic_length.py` (調和窓。4 件)、
`spikes/m36_thermodynamic_length.py`。挙動は変えていない (どこからも呼ばれない)。

測定: e77a6c5 の全セット 2 つ (TYK2 16、A2A 17、各 2 脚で 66 脚、隣接ペア 1581) の本計算 5 ns のサンプルから。GPU 無し。
出力は `spikes/out/m36/full_e77a6c5/` (追跡外)。隣接窓の距離は、両窓のサンプルでのエネルギー差の分散の平均の平方根。
ステージ長はその和。数は中央値 [5 %、95 %] で、スパイクの出力 (小数 4 桁) から丸めた。

| | complex restrain | complex discharge | complex vanish | solvent discharge | solvent vanish |
|---|---|---|---|---|---|
| ペア数 | 174 | 330 | 521 | 177 | 379 |
| 隣接窓の距離 (kT) | 1.07 [0.75, 1.65] | 0.67 [0.59, 0.98] | 1.31 [0.67, 1.77] | 1.60 [1.14, 2.22] | 1.55 [0.99, 2.32] |
| 記録された受理率 | 0.45 [0.25, 0.60] | 0.63 [0.49, 0.68] | 0.38 [0.27, 0.64] | 0.26 [0.12, 0.41] | 0.31 [0.17, 0.51] |
| 受理率 ÷ erfc(d/2) | 1.01 [0.97, 1.14] | 1.00 [0.99, 1.02] | 1.05 [0.92, 1.40] | 1.01 [0.94, 1.08] | 1.16 [0.94, 1.87] |
| 受理率 ÷ サンプルの全ペアの平均 | 1.00 [0.97, 1.03] | 1.00 [0.98, 1.02] | 1.00 [0.97, 1.03] | 1.00 [0.96, 1.04] | 1.00 [0.97, 1.03] |
| pair 重なり | 0.386 [0.283, 0.438] | 0.449 [0.402, 0.460] | 0.356 [0.297, 0.450] | 0.295 [0.196, 0.373] | 0.320 [0.230, 0.406] |
| pair 重なり ÷ ガウスの値 | 1.00 [1.00, 1.06] | 1.00 [1.00, 1.00] | 1.02 [0.98, 1.18] | 1.00 [0.98, 1.03] | 1.07 [0.97, 1.37] |
| pair 重なり ÷ A3FE の予測式 | 1.00 [0.98, 1.02] | 1.00 [0.99, 1.00] | 1.01 [0.96, 1.12] | 0.97 [0.86, 1.00] | 1.03 [0.92, 1.17] |
| pair 重なり ÷ 事前計算の値 | 0.97 [0.84, 1.10] | 0.99 [0.95, 1.02] | 0.97 [0.83, 1.09] | (stage 行列の脚) | (stage 行列の脚) |
| 上の窓の sd ÷ 下の窓の sd | 0.97 [0.66, 1.04] | 0.98 [0.91, 1.06] | 1.41 [0.92, 1.95] | 0.93 [0.85, 0.99] | 1.66 [0.81, 2.29] |
| ステージ長 (kT) | 6.2 [4.8, 7.6] | 7.3 [6.2, 8.4] | 19.3 [16.7, 22.2] | 8.7 [7.5, 9.6] | 18.7 [16.9, 20.1] |
| 今の窓数 | 6 [6, 7] | 11 [11, 11] | 16 [15, 19] | 6 [6, 7] | 12 [11, 14] |
| 受理率 0.55 での窓数 (s 0.85 kT) | 9 [7, 10] | 10 [9, 11] | 24 [21, 28] | 12 [10, 13] | 24 [22, 25] |
| 受理率 0.35 での窓数 (s 1.32 kT) | 6 [5, 7] | 7 [6, 8] | 16 [14, 18] | 8 [7, 9] | 16 [14, 17] |
| 受理率 0.27 での窓数 (s 1.56 kT) | 5 [5, 6] | 6 [6, 7] | 14 [12, 16] | 7 [6, 8] | 13 [12, 14] |
| 受理率 0.23 での窓数 (s 1.70 kT) | 5 [4, 6] | 6 [5, 6] | 13 [11, 15] | 7 [6, 7] | 13 [12, 13] |

| 脚 | 今の窓数 | 0.55 | 0.35 | 0.27 | 0.23 | 最低の受理率 | 最高の受理率 | 受理率の sd ÷ 平均 | 往復 (回/ns) |
|---|---|---|---|---|---|---|---|---|---|
| complex (33 脚) | 31 [30, 35] | 42 [36, 46] | 28 [23, 30] | 23 [21, 27] | 22 [18, 24] | 0.24 [0.11, 0.32] | 0.68 [0.64, 0.74] | 0.28 [0.23, 0.32] | 0 [0, 0] |
| solvent (33 脚) | 18 [17, 19] | 34 [32, 37] | 23 [21, 25] | 19 [18, 21] | 18 [17, 19] | 0.13 [0.08, 0.19] | 0.48 [0.37, 0.59] | 0.31 [0.25, 0.38] | 4.6 [2.2, 6.9] |

- 受理率 0.27 は pair 重なり 0.30 に、0.55 と 0.23 は A3FE の速さ 0.5 と 1.0 kcal/mol に当たる (ガウスの換算、実行の温度 298.15 K で)。
- complex 脚 33 のうち 32 は往復が 0 回 (最低の受理率は 0.06 から 0.34)。solvent 脚は 5 ns で 8 から 35 回。
- pair 重なりがガウスの値の 0.878 倍以下 (受理率 0.35 の距離に置いた窓が 0.30 以下に測られる比) のペアは 1581 のうち 4
  (complex restrain 1、complex vanish 2、solvent vanish 1)。
- vanish の長さの密度 (kT ÷ λ、λ の 10 分の 1 ごとの中央値): complex 16.7、17.1、17.6、18.2、19.3、20.9、22.1、27.7、27.7、
  17.0。solvent 12.3、13.9、(窓なし)、15.8、18.4、21.6、28.4、33.7、21.0、10.1。最大は complex が λ 0.7 から 0.9、solvent が
  0.6 から 0.8 で、端 (λ 1) では下がる (`spikes/m36_thermodynamic_length.py` の出力の最後の表。ペアの中点で 10 分の 1 に分けた)。
  complex の最後のペア (0.95 から 1) は距離 0.71、受理率 0.62 (erfc は 0.61)。
- 距離が最大のペアは A2A の complex 脚 3 本の restrain の最初 (λ 0 から 0.2): 距離 3.5 から 4.4、pair 重なり 0.12 から 0.28
  (4i は受理率 0.06)。
- ステージ長は窓数に依らなかった。ejm_31 の complex vanish (`runs/tyk2_windows_{a,b,c}_5e2cf33_{1,2,3}`): A は 18〜19 窓で
  20.65、20.31、20.06。B は 23〜24 窓で 20.01、20.45、20.20。C は 39 窓で 19.62、20.79、20.53。受理率 0.35 での complex 脚の
  窓数は 9 本とも 27 か 28 (実際は A 33〜35、B 35、C 59)。

計画への反映: 「指示の前提と違う点」の 3 (vanish 端点) は測定で決着した (発散は見えない。密度は端の手前で最大)。
既定値 (質問 1、2) はこの表を見てオーナーが決める。

短い差分レビュー (書いていない者、15 分): 窓・ステージ・λ の取り違えは無し。直したもの: この表の 7 セルの二重丸め、
受理率の関数が数でない差 (NaN) を「必ず受理」と数えていた点 (記録の数には影響なし。保存済みの u は有限)、差を
窓ごとに切り詰めていた点、端からの添字や範囲外の受理率を黙って受けていた点、長さの密度の表にスクリプトが無かった点。

### 段階 2: 記録の追加 (2026-11-25)

窓は変えない。規則も判定も長さを読まない。

- 事前計算: `schedule.yaml` の provenance に `thermodynamic_length` を足した。ステージごとに、最終の窓の `lambda`、
  `gradient_sd` (候補格子の中心差分の標準偏差)、`length` (台形則の累積)、隣接ペアの `pair_distance` と
  `exchange_acceptance` (2 窓のサンプルの全ペアの平均)。`cufep.alchemy.thermodynamic_length.stage_record`。
- 解析: 脚の結果に `neighbour_distance` (隣接窓の距離) と `stage_length` (ステージごとの和) を足した。
  `ANALYSIS_VERSION` とゴールデンの数は変えていない (固定された項目の外)。前の結果には無い (空)。
- テスト: 調和窓の事前計算で記録が既知の値になること (`test_many_initial_windows_and_a_raised_threshold_converge_by_the_overlap_of_a_pair_alone`
  に追加)、交換する調和窓 6 つの解析で距離とステージ長が既知の値になること
  (`test_the_analysis_of_a_leg_gives_the_lengths_between_its_windows`)。
- 機能確認: TYK2 のスモーク階層 (jmc_23、4 GPU、作業ツリーを載せた `cufep:dev`、`runs/benchmark_tyk2_smoke_tl_stage2`)。
  終了コード 0。両脚の `schedule.yaml` に記録があり、その `lambda` は窓と一致。結果に 2 つの数がある。スモーク階層は
  事前計算 10 ps、本計算 0.1 ns なので、数はプロトコルのものではない (complex の vanish: 事前計算の勾配から 14.8 kT、
  本計算の距離の和 20.4 kT)。
- 計画との違い: `spikes/m9_als_variants.py` での事前計算のやり直しはしていない。このスパイクは REST2 と今の既定の
  開始に追随しておらず、今の事前計算を再現しない。実パイプラインはスモーク階層で通した。既定の長さ (complex
  500 + 500 ps) の事前計算の勾配から取る長さと本計算の距離の和の比較は、段階 4 の実行で行う。
- 短い差分レビュー (書いていない者、10 分、コミット 8e41a2c): P1 と P2 は無し。事前計算の記録は共有の境界窓でも
  正しい窓・正しいステージの列を読んでいる。`stage_delta_g` は前と同じ条件で、ejm_31 の solvent 脚で保存済みの結果と
  最後の桁まで一致。`runs/` の 566 の `hrex` で、各反復が 1 状態 1 レプリカでない行は 0。直した P3 が 3 件: 同じ符号の
  無限大どうしの受理率が 1.0 だった、二乗が桁あふれする値の標準偏差が inf だった (どちらも NaN にした)、記録用の数が
  解析を止めえた (1 状態 1 レプリカでないサンプルでは長さを空にして、解析は続ける)。

### 段階 3: 新方式 (2026-11-25)

オーナーの回答 (目標受理率 0.35、床は脚の `overlap_threshold`、検証実行) を受けて実装した。既定は変えていない。

**方式が 2 回変わった。確定したのは 3 つめ。**

1. 最初の実装は Approach の手順どおり、開始の窓を残さず全部を等長に置き直した。スモーク実行で complex 脚が 6 / 11 / 15 から
   6 / 6 / 14 に減った。オーナー: 「なんで減るの？」「そもそも提案の方法は、精度が上がったんだよ / それを捨てる意味が不明 /
   それを維持しつつ実現するのが筋」。開始 A (6 / 11 / 15) は実行ごとのばらつきが最小だったので既定にした窓で、それを捨てていた。
2. 次に、開始の窓を残して長さで先に足し、その後に規則を回す形にし、「足すかどうかを重なりではなく長さで決める」と
   説明した。オーナー: 「減らさずに熱力学的長さを保つようにすることが元々の目的 / 勝手に解釈して変えている / 両方です /
   どっちかだけじゃないんだ / 趣旨に立ち返れ」。
3. 確定した方式 (Approach の冒頭の改訂): 今の規則をそのまま走らせ、その結果の窓の上に長さの条件で足す。規則と長さを
   交互に、どちらも足さなくなるまで。窓は取らない、動かさない。

実装:

- 設定: `lambda_schedule.adaptive.method` (`als` が既定、`thermodynamic_length`) と
  `thermodynamic_length.exchange_acceptance` (0.35)。どちらも 1 つの値か脚ごと。脚の規則 (`overlap_matrix` と
  `overlap_threshold`) は新方式でも今のまま (complex は pair 0.30、solvent は stage 0.18)。はじめは新方式の脚を pair に
  変えていたが、今の solvent 脚の規則を捨てることになるので戻した (オーナー: 「うまくいっていることを捨てるな」)。
  設定エラー: 刻みのガウス換算の pair 重なりが脚の閾値以下になる受理率。通知: 効かない設定。
- `adaptive_schedule`: 規則のループはそのまま。規則が足すものが無くなったら `add_by_length` が今の窓の勾配から区間の
  長さを測り、`cufep.alchemy.thermodynamic_length.windows_added_by_length` が刻みを超える区間に ceil(d / s) − 1 窓を足す
  (区間の中で λ 等間隔、候補の上。`max_windows` に当たるなら刻みが最も長い区間から 1 窓ずつ)。足されたステージは
  規則が読み直す。そのとき規則は各ペアだけの重なり (pair) を脚の閾値と比べ、不足のペアの中点に足すだけで、区間を
  等間隔に置き直さない (`add_to_pairs`。置き直すと規則が決めた窓が動く)。stage 行列の要素は窓を足すと下がるが、
  ペアだけの重なりは下がらず、その要素より小さくならない。長さが足さなくなったら終わり。規則だけで決まった窓が
  最後に全部残っていることをループの後で確かめる (残っていなければエラー)。
- 記録: `schedule.yaml` の provenance の `method` は "adaptive lambda scheduling and thermodynamic length"。
  `length_additions` に目標、刻み (kT と kcal/mol)、長さの各回 (その時点の窓、勾配の標準偏差、累積長、足した窓、
  残った最長の刻み)。規則の記録 (`iterations`) は `als` の記録がそのまま先頭に入る。`initial_windows` は開始のままで、
  結果の `lambda_windows_added` は規則と長さが足した合計。
- 前の実行の記録との比較: 設定が 2 つ増えたので、同じ YAML の再実行が「設定が違う」と扱われていた (レビューの P1)。
  `SETTINGS_ADDED` に 2 つを足し、記録の補完を入れ子の深さまで行うようにした (`write_run_record`)。
- `cufep plan` の窓の表に、新方式の脚と刻みが出る。
- 文書: `docs/configuration.md` (設定、根拠の行)、`docs/protocol.md` ("Windows added by thermodynamic length"、用語)、
  `.claude/rules/fep-invariants.md` (表の行、判定なしの数の一覧)。

テスト (調和窓、答えが分かっている系):

- 26 幅の直線の経路。閾値 0.20 (pair) と 0.18 (stage 行列、今の solvent 脚の規則)、2 幅おきの 14 窓: 規則は足さず
  (`als` は 14 窓)、新方式は全区間に 1 窓足して 27 窓。14 窓はすべて含まれる。41 窓 (刻みの内) からは何も足さず、窓も
  規則の記録も `als` と同じ。
- 端で急な経路 (6λ + 20λ⁴) と、等間隔 11 窓が解像しない 6 幅の段差: どちらも規則が先に足し、その窓の上に長さが足す。
  `als` の窓はすべて新方式の窓に含まれ、`als` の規則の記録は新方式の記録の先頭と一致し、最後の記録は両条件を満たす
  (全ペアの pair 重なりが 0.30 を超え、最長の刻みが 1.32 以下)。
- 規則が窓を動かさずに足す関数 (`add_to_pairs`)、区間ごとの窓数 (`windows_added_by_length`)、設定、前の実行の記録。
- GPU: 実際のワーカーのプールで両方式 (`test_precalculation_windows_run_in_a_pool_of_workers`、両バックエンド)。

機能確認: TYK2 のスモーク階層の設定に新方式を足した実行 (jmc_23、4 GPU、作業ツリーを載せた `cufep:dev`、最終のツリー、
`runs/tl_smoke_stage3c`): 終了コード 0。complex (pair 0.30) は開始 6 / 11 / 15 から規則が vanish に 2、長さが restrain に 1 と
vanish に 3 足して 7 / 11 / 20 (38 窓)。solvent (stage 0.18) は 5 / 9 から規則が 7 / 17 まで足し (今の方法と同じ規則)、長さが
3 / 3 足して 10 / 20 (29 窓)。どの脚も窓は減らず、最後の記録は両条件を満たす。スモーク階層の事前計算は 10 ps なので、
足された数はプロトコルのものではない。

保存済みの実行からの見込み (`spikes/m36_thermodynamic_length.py`、中央値 [5 %、95 %]): 今の窓をすべて残して刻み 1.32 kT で
足すと、complex 脚は 31 [30, 35] から 40 [38, 45] (vanish +8 [6, 10]、restrain +1 [0, 2]、discharge +0)、solvent 脚は
18 [17, 19] から 30 [28, 33]。刻み 1.56 kT (受理率 0.27) なら 37 [31, 40] と 27 [24, 30]。

短い差分レビュー (書いていない者、段階 3 で 2 回): 既定 (`als`) の動作は HEAD と同じ。P1 は 1 件: 設定が増えたことで、同じ YAML の
再実行が前の記録と「違う」と扱われた (上で修正、回帰テストあり)。規則の読み直しが規則の窓を動かしうる点 (P2) は
`add_to_pairs` とループ後の確認で構造的に塞いだ。`max_windows` の注記がステージごとでなかった点 (P3) も直した。

### 段階 4: 検証実行 (2026-11-25)

どの実行もイメージ `cufep:eeb2351`、ejm_31、8 GPU、本計算 5 ns/窓。数は kcal/mol、± は標準誤差。

コードと記録の照合 (GPU なし、`spikes/out/m36_scratch/`、追跡しない):

- TIP3P、e77a6c5 と eeb2351: 準備済みの System (両脚)、alchemical System (同じ入力から両方のコードで作り、バイト単位)、
  ワーカーの設定 (違いは新しい空欄 `gcmc`、`relocation` だけ)、拘束の原子と力の定数、同じサンプルの解析が同じ。
- OPC、29fb9df と eeb2351: 準備済みの System と箱 (両脚。今のコードで準備し直して比べた)、alchemical System、ワーカーの
  設定、解析が同じ。
- A2A 4e、e77a6c5 と eeb2351: alchemical System (両脚)、ワーカーの設定、解析が同じ。solvent 脚の準備済み System も同じ。
  膜の準備は同じコードでも毎回違う: OpenMM の `Modeller.addMembrane` が種を与えない Langevin 積分器で膜を馴染ませるため。
  保存済みは 73,274 原子 (重なりで除いた水 320)、e77a6c5 のコードで準備し直すと 73,033 (脂質 90 + 90、水 299)、eeb2351 では
  73,295 (91 + 91、315)。TIP3P では膜のコードの差分 (4 サイトの水のための分岐と記録の `water`) は通らない。
- 準備の座標 (水とイオンの置き方) は同じイメージの実行同士でも毎回違う (TIP3P の 5 実行で `complex.pdb` がすべて別)。
- イメージの違い: cupy と CUDA の開発用ライブラリが加わり、scipy が 1.18.1 から 1.17.1 になった。

TIP3P (実験値 -9.54):

| 実行 | イメージ | 方式 | ΔG | complex 脚 | solvent 脚 | 窓 | 最低重なり | 往復 |
|---|---|---|---|---|---|---|---|---|
| 対照 `tyk2_als_eeb2351_1` | eeb2351 | als | -9.88 ± 0.66 | 40.90 ± 0.63 | 20.46 | 33 / 17 | 0.154 / 0.163 | 0 / 14 |
| V1、V2 `tyk2_tl_eeb2351_{1,2,3}` | eeb2351 | thermodynamic_length | -11.48 ± 0.32、-11.58 ± 0.32、-11.64 ± 0.45 | 42.31、42.27、42.30 | 20.25、20.10、20.08 | 38 / 30、39 / 30、39 / 28 | 0.146、0.148、0.140 / 0.160、0.159、0.161 | 0 / 44、50、37 |
| start A `tyk2_windows_a_5e2cf33_{1,2,3}` | 5e2cf33 | als | -9.36、-9.63、-9.33 | 39.97、40.30、40.35 | 20.04、20.12、20.47 | 35 / 17、33 / 18、35 / 17 | 0.141、0.153、0.147 / 0.191、0.165、0.130 | 1、0、0 / 22、16、9 |
| 全セット `benchmark_tyk2_full_e77a6c5` | e77a6c5 | als | -10.08 ± 0.37 | 40.43 | 19.79 | 33 / 18 | 0.154 / 0.189 | 0 / 21 |

- 同じイメージで、新方式の complex 脚は対照より +1.37 から +1.41、solvent 脚は -0.21 から -0.38。
- 対照の complex 脚は、11 月 22 日までのイメージで今の方式を走らせた 4 回より 0.47 から 0.93 高い。
- 状態ごとの自由エネルギーを対照と比べると、新方式の 3 回は終点でどれも +1.35 から +1.40。差がつく場所は実行ごとに違い、
  1 回目は discharge (その終わりまでに +1.5)、2 回目と 3 回目は vanish の λ_sterics 0.9 から 0.15。
- 新方式の実行を、対照にある窓だけで推定しても値は変わらない (全窓 37.67、37.82、37.90 に対して共通の窓だけで 37.64、
  37.80、38.12。同じ計算で対照は 36.43)。足した窓が推定を変えたのではなく、共通の窓でサンプルされた配置が違う。

OPC (実験値 -9.54):

| 実行 | イメージ | 方式 | ΔG | complex 脚 | solvent 脚 | 窓 | 最低重なり | 往復 |
|---|---|---|---|---|---|---|---|---|
| 対照 `tyk2_als_opc_eeb2351_1` | eeb2351 | als | -8.02 ± 0.77 | 40.72 ± 0.72 | 22.15 ± 0.28 | 34 / 21 | 0.150 / 0.185 | 0 / 9 |
| 新方式 `tyk2_tl_opc_eeb2351_1` | eeb2351 | thermodynamic_length | -8.97 ± 0.44 | 40.83 ± 0.36 | 21.34 ± 0.26 | 36 / 32 | 0.139 / 0.160 | 0 / 29 |
| 全セット `benchmark_tyk2_full_opc_29fb9df` | 29fb9df | als | -9.30 ± 0.57 | 40.90 ± 0.55 | 21.05 ± 0.14 | 35 / 19 | 0.156 / 0.182 | 0 / 10 |

- 同じ方式と同じ System で、solvent 脚が 1.09 違う。そのうち 0.83 は discharge で、0.43 は最初の窓の組 (λ_elec 1.0 と 0.92)
  で生じている。完全に結合した窓の統計的非効率は対照で 23 反復、29fb9df の実行で 178 反復。
- 同じイメージで、新方式の complex 脚は対照より +0.11 (TIP3P の +1.4 と違う)、solvent 脚は -0.81。新方式が complex 脚に
  足した窓は 2 (TIP3P では 5 から 6)。solvent 脚の差は 29fb9df の実行と同じ形で、最初の窓の組で -0.39、discharge の終わり
  までに -0.58。完全に結合した窓の統計的非効率は新方式で 96 反復。
A2A 4e (膜、実験値 -12.07):

| 実行 | イメージ | 方式 | ΔG | complex 脚 | solvent 脚 | 窓 | 最低重なり | 往復 |
|---|---|---|---|---|---|---|---|---|
| 対照 `a2a_als_eeb2351_1` | eeb2351 | als | -14.39 ± 0.61 | 40.75 ± 0.60 | 16.53 ± 0.12 | 31 / 18 | 0.162 / 0.183 | 0 / 27 |
| V3 `a2a_tl_eeb2351_1` | eeb2351 | thermodynamic_length | -15.02 ± 0.37 | 41.43 ± 0.36 | 16.60 ± 0.09 | 32 / 29 | 0.165 / 0.156 | 0 / 57 |
| 全セット `benchmark_a2a_full_e77a6c5` | e77a6c5 | als | -13.50 ± 0.34 | 39.76 ± 0.32 | 16.54 ± 0.12 | 30 / 17 | 0.169 / 0.178 | 0 / 22 |

- 対照は最初の試みで、背景実行の時間制限 (30 分) によりラッパーごと complex 脚の本計算の反復 840 で止まり、同じコマンドで
  チェックポイント (反復 800) から再開した (ログに記した)。
- 拘束はどちらも PHE168 の FaceToFace (スコア 106.1)。PHE168 の原子の順が N, CA, C と N, C, CA で違い、拘束を外す補正は
  -9.83 と -9.72。complex 脚の最初の窓 (拘束なし、完全に結合) の統計的非効率は 203 反復と 415 反復。
- V3: complex 脚は規則が開始 6 / 11 / 15 の restrain に 1、長さが vanish に 1 足して 32 窓。solvent 脚は長さが 12 足した。
  拘束は GLU169 の HBDonor (スコア 105.6) で、対照 (PHE168 の FaceToFace) と違う: 拘束は実行ごとの MD から選ばれる。
  complex 脚の最初の窓の統計的非効率は 9 反復。同じイメージで対照より complex 脚 +0.68、solvent 脚 +0.07。

同じイメージで新方式から対照を引いた差 (kcal/mol):

| 系 | complex 脚 | solvent 脚 | ΔG |
|---|---|---|---|
| ejm_31 TIP3P (3 回) | +1.37 から +1.41 | -0.21 から -0.38 | -1.60 から -1.76 |
| ejm_31 OPC | +0.11 | -0.81 | -0.95 |
| 4e (膜、TIP3P) | +0.68 | +0.07 | -0.63 |

「Verification」の指標 (`spikes/m37_method_verification.py`、出力 `spikes/out/m37/`; 古い解析の結果には隣の窓との長さが
ないので、本計算のサンプルから同じ関数で求めた。両方ある実行で差は 0):

| 系 | 実行 | 受理率 0.35 未満の組 complex / solvent | 1.32 kT より長い組 complex / solvent | 往復 complex / solvent | 計算量 ns |
|---|---|---|---|---|---|
| ejm_31 TIP3P | 対照 | 8 / 32、13 / 16 | 9、13 | 0 / 14 | 337 |
| | 新方式 1、2、3 | 1 / 37、1 / 38、0 / 38; 2 / 29、0 / 29、0 / 27 | 3、0、1; 2、1、2 | 0 / 44、50、37 | 450、457、443 |
| | 今の方式 (5e2cf33 × 3、e77a6c5) | 5 から 6 組; 11 から 13 組 | 6 から 9; 12 から 14 | 0 から 1 / 9 から 22 | 342 から 350 |
| ejm_31 OPC | 対照、新方式、29fb9df | 8、4、7; 11、2、13 | 9、5、8; 12、2、14 | 0 / 9、29、10 | 369、448、364 |
| A2A 4e | 対照、新方式、e77a6c5 | 2、4、6; 10、0、11 | 8、8、9; 10、1、13 | 0 / 27、57、22 | 326、400、313 |

- 窓: 長さが足したのは complex 脚の vanish (TIP3P で 3 から 5、OPC 2、4e 1) と restrain (0 か 1)、solvent 脚 (11 から 15)。
  修復 (長さの後に規則が足した窓) はどの実行でも 0。
- 本計算の前半と後半だけでの complex 脚 (ejm_31 TIP3P): 新方式 3 回の平均 42.50 と 42.05、今の方式 5 回の平均 40.04 と 40.72。
  差は前半 +2.46、後半 +1.33。
- 同じ計算量 (各脚の本計算を 対照の窓 × 5 ns ÷ 新方式の窓 に切った): ejm_31 TIP3P の ΔG -11.19 ± 0.43、-11.67 ± 0.51、
  -11.98 ± 0.44 (complex 42.27、42.40、42.69)、OPC -8.31 ± 0.52、4e -14.92 ± 0.53。対照との差は -1.31 から -2.10、-0.29、-0.53。
- 報告のページ (図と表): https://claude.ai/artifact/PhX8DDrUoLXj7yE8yCrVrJ (非公開)。

既定は変えていない (`method: als`)。既定を変えるかはオーナーの判断。

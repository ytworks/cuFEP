# λ スケジュール: 熱力学的長さによる配置と重なりの床
Status: Planned (2026-11-25: 計画のみ承認。実装は未着手。「質問」7 件はオーナーの判断待ち)
Refs: this pull request; `docs/protocol.md` ("Adaptive lambda scheduling"); `docs/configuration.md` (`lambda_schedule.adaptive`); `docs/plans/2026-11-21-als-initial-window-design.md`

## Context

オーナーの指示 (2026-11-25): λ スケジューリングを、現行の ALS (重なり閾値による逐次挿入、Midgley et al.,
JCIM 2025, 65, 512) から、熱力学的長さによる配置 (A3FE: Clark et al., JCTC 2024, 20, 7806) と重なりの保証を
両立する方式に拡張する。この計画は開発プランで、実装は含まない。既定の挙動は変えず、新方式は opt-in。

要点:

- 新方式は「提案の窓 → 熱力学的長さで等間隔に置き直す → 既存の ALS 規則で外れた窓だけ修復」の3手。修復は今の
  `als_step` のループをそのまま使う。
- 計量は、事前計算が既に全候補 (101 点) で計算しているエネルギーの中心差分から取る。追加のエネルギー計算も
  ∂H/∂λ の出力も要らない。
- s は「隣接窓の交換受理率の目標」から決める。目的は窓数 (= GPU 時間) の最小化、制約は受理率の目標と pair
  重なりの床。往復率と実効サンプル数は報告する数にとどめる。
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
3. **vanish 端点で σ² が発散するかは未確認。** cuFEP のソフトコアは Beutler 形 (α 0.5)。段階 1 で測る。

既知の実測 (`docs/plans/2026-11-21-als-initial-window-design.md`): complex 脚の往復は 13 実行すべてで 0〜1 回で、
最低隣接受理率が 0.07 でも 0.47 でも変わらない。受理率が最も高い C が ΔG のばらつきも最大だった。

## Approach

### 手順 (脚ごと、ステージごと)

1. **提案。** `initial_windows` の窓を今と同じ事前計算で走らせる (変更なし)。
2. **計量。** 各提案窓で σ = sd[(u(λ+Δλ) − u(λ−Δλ)) / 2Δλ] を自分のサンプルから取る (Δλ = 0.01、端は片側差分)。
   候補格子上で線形補間して積分し、累積長 Λ(λ) とステージ長 L を得る (kT)。A3FE 式 19 (g = 1) と同じ量。
3. **配置。** s = 2·erfcinv(a\*)、区間数 K = ceil(L/s)。Λ が等間隔になる候補に窓を置く (両端は固定、s は脚の
   全ステージで共通)。窓は提案より減ってよい。合計が `max_windows` を超えるなら L に比例して K を減らし、注記を残す。
4. **実測と修復。** 既存のループに戻る。新しい窓だけ走らせ、`pair_overlap` を測り、脚の `overlap_threshold` 以下の
   区間に `als_step` が 1 窓ずつ足す。停止条件と注記は今のまま。
5. **記録。** `schedule.yaml` の provenance に、方式、a\*、s (kT と kcal/mol)、ステージごとの L と σ、提案の窓、
   配置した窓、ペアごとの予測と実測、修復の反復を書く。

組み合わせ (A3FE の計量、プレプリントの受理率目標、Midgley の規則での修復) は cuFEP 独自として文書に記す。

ガウス近似での換算 (手計算。段階 1 の単体テストで固定する):

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
| 1 | 測定の関数 (新モジュール `src/cufep/alchemy/thermodynamic_length.py`: 勾配の sd、累積長、受理率の予測、ガウス換算)。スパイク `spikes/m35_thermodynamic_length.py` で保存済み 66 脚の本計算 u から、ステージ長、ペアごとの d、ガウス予測対実測、s ごとの予測窓数、事前計算と本計算の pair 重なりの差を出す。挙動は変えない | 単体テスト: 調和窓で長さ = 両端の距離、受理率 = erfc(d/2)、既存の固定値 0.400 / 0.223。既存テスト全通過。GPU 不要。表を報告 |
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
| vanish 端点で σ が大きく、窓が端に集まるか `max_windows` に当たる | V0 のステージ長。実行時は注記 |
| 設定の食い違い (目標が床より下、stage の明示) | 設定エラーとそのテスト |
| 入口による違い (再開、`analyze`、`freeze`、`--reschedule`) | 段階 3 の入口横断テスト |
| 窓が必要以上に近く、解析の stage 行列が 0.03 を割る | 換算表では d ≥ 0.11 (a\* ≤ 0.94) で起きない。V1 の `min_overlap` |

## 質問と未確認

質問 (オーナーの判断):

1. 目標受理率の既定。0.35 (査読前のプレプリント) を仮に置き、V0 の表を見て決める進め方でよいか。査読前の値を
   根拠にできないなら、A3FE の速さ 0.5〜1.0 kcal/mol (受理率換算 0.55〜0.23) から選ぶ。
2. 修復の床。新方式でも脚の `overlap_threshold` (pair 0.30) を使う案でよいか。
3. 検証実行は ejm_31 を 3 回と 4e を 1 回でよいか。
4. `--reschedule` は今と同じ (事前計算からやり直す) にとどめる。前回の本計算の u を計量の入力に使う拡張は要るか。
5. 事前計算の候補エネルギーは保存せず、導いた曲線だけ `schedule.yaml` に書く案でよいか (生データは complex 脚で
   数十 MB)。
6. 「Opt-PSO (Lee et al.)」は Zhang, Giese, Lee, York の論文を指すか。
7. 検証は solvent 脚も新方式にする案でよいか。

未確認 (作業で埋める):

- A3FE の SI (S1、S2、S10)、Minh、Rizzi、Nguyen & Minh、Zhang et al. の全文 (段階 0)。計画の時点で読んだのは
  A3FE 本文の 2.1、3.3、4.2.1 節とプレプリントの S1.8 節。
- cuPotential のパラメータ微分への対応 (計画は依存しない)。
- vanish 端点での σ の振る舞い (V0)。
- 換算表の数値 (段階 1 の単体テスト)。

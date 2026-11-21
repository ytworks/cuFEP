# adaptive lambda scheduling の初期 window を設計し直す (max_windows 100、3 案を TYK2 example で評価)
Status: In progress (手順 1〜3 済み: 計画と draft PR #7、`max_windows` 100、初期 window を数または値の列で)
Refs: this pull request; issue #5; `docs/protocol.md` ("Adaptive lambda scheduling"); `docs/configuration.md` (`lambda_schedule`)

## Context

TYK2 example (ejm_31) を同じ YAML で 4 回実行した結果は -12.16、-12.23、-8.84、-8.33 kcal/mol
(標準偏差 2.09) で、差は complex leg (39.34〜42.95、標準偏差 1.86) にある。1 回の run が報告する
標準誤差 (0.3〜0.9) はこれより小さい。原因は確定していない (issue #5、`PROGRESS.md`)。

オーナーの指示と決定 (2026-11-21):

1. adaptive lambda scheduling (ALS) の `max_windows` を 100 に上げる。
2. ALS は必須 (分子ごとに window を変える手間を省く)。そのうえで仮説: 初期 window はある程度
   増やすべきで、等間隔である必要もない。どこを細かく、どこを粗くするかの設計が重要。
3. threshold も上げる。そのために ALS の不足の判定を、隣り合う 2 window だけから求めた overlap に
   変える。今の判定 (stage の全 window の overlap 行列の隣どうしの要素) は、重なる window の数で
   値を分け合うので window が近いほど下がり、密な初期 window では足す必要のない所に window を
   足す (`docs/protocol.md` "Why the threshold cannot go much higher")。
4. window の設定は complex leg と solvent leg で別のものを動かせるようにし、ユーザーも YAML で
   設定できるようにする。
5. バラツキを抑えつつ計算時間も抑える既定値を 3 案作り、TYK2 example を各 3 回実行して評価する
   (9 run)。この計画の承認が評価 run の開始までの許可になる。
6. draft pull request を作り、こまめに push する。

調べて分かった事実 (cuFEP、`main` の 4e0b8c4):

- window 数の上限は ALS の `max_windows` (complex 48 / solvent 32) だけ。fixed mode、replica
  exchange、保存、scheduler に上限はない。threshold 0.18 では TYK2 の complex leg は 21〜25 window
  で収束し (full benchmark でも最大 30)、上限に届いた run はない。
- `initial_windows` は stage ごとの数 (等間隔) だけを受け取る。`overlap_threshold`、`candidates`、
  `mode` は両方の leg に共通の 1 つの値。`window_ps` と `equilibration_ps` は 1 つの値または
  leg ごとの値 (`PrecalculationLengths`、`AdaptiveConfig.window(leg)`)。
- ALS の設定を変えて同じ出力先でもう一度実行しても、`schedule.yaml` があれば ALS はやり直さない
  (`--reschedule` でやり直す)。

### 設計の根拠 (保存済みの run から読んだ値。新しい計算はしていない)

full benchmark の 2 run (commit 3b45bfe、TYK2 16 リガンド、A2A 17 化合物、ALS 0.18、
production 5 ns) の全 leg を lambda の 10 分の 1 ごとに集計した。

ALS が window を置いた密度 (complex leg の vanish stage、区間の数の平均):

| lambda | 0〜0.5 | 0.5〜0.6 | 0.6〜0.7 | 0.7〜0.8 | 0.8〜0.9 | 0.9〜1.0 |
|---|---|---|---|---|---|---|
| TYK2 | 各 0.7〜1.4 | 1.4 | 1.7 | 1.9 | 1.9 | 1.2 |
| A2A | 各 0.2〜1.0 | 1.1 | 1.0 | 1.4 | 1.6 | 1.1 |

OpenFE の推奨値 (0.6 以降が 2 倍密) とほぼ同じ形。restrain stage は lambda の小さい側 (0〜0.33) に
足され、discharge stage はほぼ等間隔のまま。

遅さ (window ごとの統計的非効率 g の幾何平均、かっこ内は最大。1 iteration = 2.5 ps):

| complex leg の vanish | 0.1〜0.6 | 0.6〜0.7 | 0.7〜0.8 | 0.8〜0.9 | 0.9〜1.0 |
|---|---|---|---|---|---|
| TYK2 | 1.5〜2.1 | 3.6 | 9.2 (44) | 16.8 (161) | 5.4 (234) |
| A2A | 1.4〜2.7 | 4.2 | 21.8 (174) | 33.5 (481) | 17.3 (143) |

- 独立サンプルが 50 未満の window は、complex leg では大半が vanish の 0.7〜1.0 にある
  (TYK2 9 個中 7、A2A 32 個中 24)。2 つの標的で同じ場所。残りは restrain stage にあり、
  A2A ではその 6 個が最初の window。
- ほかに遅いのは leg の最初の window (結合した状態そのもの): complex の restrain 0.0 で
  g 6.0 (TYK2)、26 (A2A)、solvent の discharge 0.0 で 12.7 (TYK2)。それ以外はおおむね 1〜6。
- 今の ALS は overlap の必要には応えているが、遅い区間 (vanish 0.7〜0.9) の window は overlap が
  足りた時点で止まるので、そこに置かれる replica は 3〜5 個。

設計の考え方: 区間ごとの分散はおよそ (Δλ)² × σ² × g に比例する。window の総数が同じなら、
密度を「overlap から要る密度 × g の平方根」にすると合計が最小になる。上の表から、vanish stage は
window の約半分が 0.7〜0.9 に入る配分になる (TYK2 49 %、A2A 51 %)。この配分は cuFEP 独自の
設計として記録する (実装の前に、同じ考え方の文献を原典で確認する)。

## Approach

### 1. コードの変更 (評価が終わるまで、既定の挙動は `max_windows` 以外変えない)

- `lambda_schedule.adaptive.max_windows` の既定値を complex 100 / solvent 100 にする
  (`WindowLimits` in `src/cufep/config/schema.py`、`AlsSettings.max_windows` in
  `src/cufep/alchemy/schedule.py`)。
- `lambda_schedule.adaptive.initial_windows` を、stage ごとに「数 (等間隔、今のまま)」または
  「lambda の値の列」で書けるようにする。`lambda_schedule.windows` と同じ書き方 (`StageWindows`、
  `_check_stage`)。値は candidate (0.01 刻み) の上にあることを設定の段階で確かめ、外れていれば
  何も書く前に断る。`adaptive_schedule`、前計算の worker の数、provenance、
  `lambda_windows_added` が列を扱えるようにする。
- 新しい設定 `lambda_schedule.adaptive.overlap_estimate: stage | pair` (名前は実装時に確定)。
  - `stage`: 今の判定 (論文のとおり、stage の全 window の行列)。評価が終わるまで既定のまま。
  - `pair`: 隣り合う 2 window のサンプルだけで作った 2 状態の overlap 行列の要素 (0〜0.5)。
  - `stage_overlap` の隣に置き、解析と同じ `cufep.analysis.mbar.overlap_matrix` を使う
    (2 window の stage では `stage` と同じ値になる作り)。
  - iteration の記録の `superdiagonal` は今のまま stage の行列の値を残し (既存の script と test が
    読む。production の基準と比べる値でもある)、`pair` のときは `pair_overlap` を足す。
    provenance と log と `cufep` の表示に判定の種類を出す。キーのない古い `schedule.yaml` は
    `stage` と読む。
  - `workflow.py` で `AlsSettings` を組み立てている 1 か所を `als_settings(cfg, leg)` に切り出す。
    leg ごとの値の読み分けをここに集め、CPU の test で確かめられるようにする。
- window を決める設定を、どれも「両方の leg に 1 つの値」または「leg ごとの値」で YAML に
  書けるようにする。今ある `window_ps` / `equilibration_ps` と同じ書き方にそろえる。

  | 設定 | 今 | 変更後 |
  |---|---|---|
  | `lambda_schedule.mode` | 1 つ | 1 つ、または `{complex, solvent}` |
  | `adaptive.overlap_threshold` | 1 つ | 1 つ、または `{complex, solvent}` |
  | `adaptive.overlap_estimate` (新) | - | 1 つ、または `{complex, solvent}` |
  | `adaptive.candidates` | 1 つ | 1 つ、または `{complex, solvent}` |
  | `adaptive.initial_windows` | leg ごと (数) | leg ごと (数または値の列) |
  | `adaptive.max_windows` | leg ごと | そのまま |
  | `windows` (fixed mode) | leg ごと | そのまま |

  leg の job は自分の leg の値だけを読む。1 つの値で書いた今の YAML はそのまま通り、
  `resolved_config.yaml` と文書の既定値は 1 つの値の書き方のまま。
- 挿入の規則 (最長の bad section に 1 window、等間隔に置き直す)、candidates 101、前計算の長さは
  変えない。結果の解析の基準 (行列で 0.03 以上) も変えない。
- 100 window でも replica exchange、保存、MBAR、図、表示が通ることを確認する。コードを読んだ
  限り上限はない。解析の計算量は window 数のほぼ 3 乗で増える (推定 1 回が K² × サンプル数、
  サンプル数が K に比例)。実測してオーナーに報告し、直すかは別に決める。
- この pull request は既定値を変える commit まで merge しない。上限 100 と今の判定 `stage` の
  組み合わせは、収束しない leg が 48 / 32 でなく 100 まで足すため。
- `schedule.yaml` がある leg で ALS をやり直さない挙動は変えない。評価は run ごとに新しい出力先を
  使う。

### 2. 3 案 (example の YAML に上書きして評価する)

共通: ALS あり、`max_windows` 100 / 100。それ以外は example のまま (equilibration 1 ns、
production 5 ns、REST2、restraint)。

- complex leg: `overlap_estimate: pair`、threshold 0.30。3 案の違いは初期 window の設計だけ。
- solvent leg: 3 案とも今の設定のまま (等間隔 5 / 9、threshold 0.18、判定 `stage`)。run 間の差は
  complex leg にあり (標準偏差 1.86 に対して solvent 0.29)、solvent leg は最初の window 以外に
  遅い区間がない (g 約 1.1) ため。leg ごとの設定はこの形で使う。

| 案 | complex leg の初期 window の設計 | 初期の数 (complex) | ALS 後の見込み (complex) |
|---|---|---|---|
| A: 文献の形 | OpenFE の推奨値。restrain 6 と discharge 11 は等間隔、vanish 15 は 0.6 以降が 2 倍密 | 30 | 約 37〜41 |
| B: 設計した形 | overlap の必要 × 遅さの平方根。vanish の約半分を 0.7〜0.9 に置く。数は A と同じ | 30 | 約 33〜36 |
| C: 設計した形を 2 倍 | B と同じ配分で数を 2 倍 | 59 | 約 59〜62 |
| (今) | 等間隔 3 / 5 / 9、threshold 0.18 (判定 `stage`) | 15 | 21〜25 (実測) |

ALS 後の見込みはガウス分布の近似からの値で、実測ではない。B の complex leg の目安 (上の表から
手で出した値。最終値は集計の script が出す):

- restrain 7: 0, 0.08, 0.16, 0.25, 0.33, 0.67, 1.0
- discharge 7: 等間隔
- vanish 18: 0, 0.15, 0.30, 0.42, 0.52, 0.60, 0.66, 0.71, 0.74, 0.77, 0.80, 0.82, 0.84, 0.86,
  0.88, 0.90, 0.95, 1.0

何が分かるか:

- 今 → A: 初期 window を増やす効果 (形は文献)。
- A → B: 同じ数で、遅い区間に寄せる設計の効果。
- B → C: 同じ設計で、数を増やす効果。

threshold を 0.30 にそろえる理由: pair の判定では 0.18 が今の密度に当たり、0.30 は難しい区間で
約 1.5 倍の密度を要求する。0.40 は candidates 101 (0.01 刻み) では vanish 0.75〜0.88 で届かない
run が出る見込みで、`saturated` の警告が付く。

production の長さは案に入れない。5 ns で走らせ、短くした場合の値は同じ run のサンプルの前半から
0.5 ns 刻みで出す。既定の `production_ns` は評価の表からオーナーが選ぶ。

### 3. 評価

- イメージ: push した commit の clean な checkout から `cufep:<commit>` を作る。
- run: 案ごとに 3 回、同じ YAML と seed (今の 4 回と同じやり方)。GPU 4 枚ずつ 2 run を並行、
  `spikes/run_frozen.sh`、出力は `runs/tyk2_windows_{a,b,c}_<commit>_{1,2,3}` (untracked)。
- 順序: 各案の 1 回目を先に走らせ、window 数、ALS が足した数と場所、警告 (`max_windows`、
  `saturated`) を報告する。異常があれば止めて報告する。なければ 2・3 回目に進む。
- 比較の基準: 既存の 4 run (`runs/tyk2_interactions_720cdb1`、`runs/tyk2_spread_1b770da_{1,2,3}`)。
  sampling のコードは変えないので、今の設定での再実行はしない。
- 報告する数値 (判定の言葉は付けない):
  - run ごと: ΔG_bind ± 標準誤差、complex leg、solvent leg、restraint の解放、実験値 -9.54 との差。
  - 案ごと: 3 回の平均、標準偏差、最大と最小の差。production を 0.5 ns 刻みで短くした場合の同じ値。
  - window: stage ごとの数、ALS が足した数と場所、overlap の最小値 (基準 0.03)、独立サンプルの
    最少 (基準 50) とその window の場所、round trip。
  - 実行時間はオーナーへの報告と untracked の出力だけに書く (`docs/LICENSING.md`)。
- 集計は `spikes/` の script。production を短くした場合の値は、保存したサンプルを前から切って
  パッケージの関数に渡して出す (`leg_result_from_samples`、`combine_ligand`)。短くした場合も
  overlap と独立サンプルの確認つきの値になる。
- 上の「設計の根拠」の表と B・C の値を出す script も `spikes/` に置く。出力は `spikes/out/`
  (untracked)。

### 4. 評価のあと

オーナーが案と production の長さを選ぶ。そのあとで既定値 (判定 `pair`、初期 window、threshold、
必要なら `production_ns` と example の YAML) をまとめて変え、`docs/configuration.md` (根拠の表を
含む)、`docs/protocol.md`、`.claude/rules/fep-invariants.md` の表、`PROGRESS.md` を同じ commit で
直す。A2A example と benchmark の再実行はこの計画に入れない (not run)。

### 5. 手順

進め方 (全手順に共通):

- 最初の commit (この計画) を push した時点で、main への draft pull request を作る。既定値を
  変える最後の commit まで draft のまま置く。
- こまめに push する: 下の手順 2〜6 はそれぞれ 1 つ以上の小さい commit にし、その手順の test が
  通るたびに commit して、すぐ push する。評価の結果 (手順 9) も、案ごとに出た時点でこの計画に
  書いて push する。
- 各手順の前と各 push のあとに pull request の review を読む。新しい指摘は、次の手順より先に、
  再現してから直す。
- push のたびに CI (`unit`、`harness`) の結果を見る。失敗したまま次の手順に進まない。

1. branch `feat/als-initial-window-design`、この計画、draft pull request。
2. `max_windows` 100 / 100: schema、`AlsSettings`、`docs/configuration.md`、`docs/protocol.md`、
   48 / 32 を固定している test、`spikes/m9_als_variants.py`。
3. `initial_windows` を数または値の列に。
4. `overlap_estimate` の設定と `pair` の実装。`mode`、`overlap_threshold`、`overlap_estimate`、
   `candidates` を 1 つの値または leg ごとの値に。
5. 100 window の確認 (CPU の test 1 本、必要なら修正)。
6. 集計の script で「設計の根拠」の表と B・C の初期 window を出し、3 案の YAML を作る。
7. 検証 (下) を全部通す。diff 全体を書いていない読み手にレビューさせ、指摘を直して push する。
   Codex の review を pull request に付ける。指摘は再現してから直す。
8. 評価 (上の 3)。
9. 結果をこの計画と `PROGRESS.md` に書き、オーナーの選択を待つ。
10. 選ばれた案で既定値を変える (上の 4)。

### オーナーに先に伝えた点

- 3 回の標準偏差は粗い (95 % の幅は値の約 0.5〜6 倍)。今の 2.1 から 1 未満への大きな変化は
  見えるが、案どうしの小さな差は 3 回では区別できない。window の場所ごとの独立サンプル数は
  1 回の run でも比べられるので、あわせて報告する。
- `docs/protocol.md` は「13〜34 window では run 間の差は変わらなかった」と記録している。ただし
  どれも overlap に合わせた配置で、遅い区間に寄せた配置は測っていない。
- 4 run の差は restrain、discharge、vanish の 3 stage でほぼ同じ向きに出ている (平均からの
  ずれの符号が 12 個中 11 個そろう。標準偏差は順に 0.36、0.48、1.21 kcal/mol)。window の配置では
  動かない、run 全体に共通する原因がある場合、3 案とも差は縮まらない。そのときもその数値を
  報告する。
- production を短くすると「window ごとに独立サンプル 50 以上」(Mey et al. 2020) を満たさない
  window が増える。5 ns の今でも 8 leg 中 5 leg にある。
- 設計の根拠は `restraints.selection: auto` の benchmark (33 leg)。今の既定 `interactions` の
  4 run では、restrain の終わりにも遅い window (g 約 100) が 2 run にある。集計には両方を入れる。

## Invariants

- 判定の変更 (`pair`) と初期 window の設計は論文の方法からの変更で、オーナーの決定
  (2026-11-21)。`.claude/rules/fep-invariants.md` の表の ALS の行 (threshold と、何の overlap か) と
  `docs/configuration.md`、`docs/protocol.md` は、既定値を変える commit で出典つきで直す。評価の
  間は既定値を変えない。
- 「目玉機能を、入っているが何もしない設定にしない」: run ごとに ALS が足した window の数と
  場所を報告する。足さない leg が多い案はそのとおり報告する。
- 判定の言葉を付けない。基準を run を通すために緩めない。結果の解析の基準 (overlap 0.03、
  独立サンプル 50) は変えない。
- 解析の数値は変わらない (`ANALYSIS_VERSION` はそのまま、golden も変えない)。
- 1 リガンド 1 回が既定のまま。3 回は protocol の検証のための測定。
- cuPotential の速度と実行時間は tracked なファイル、commit message、GitHub に書かない。

## Verification

- CPU: `scripts/lint.sh`、`scripts/in_docker.sh python -m pytest -q`、
  `check_progress_size.py`、`check_readme_pair.py`、`check_names.py`。
- 新しい test (`tests/unit/test_schedule.py`、`test_config.py`):
  - 調和ポテンシャルの window (幅 1、長さ 8 を間隔 1、0.5、0.25 で) で、間隔を詰めると pair の
    overlap は上がり、行列の要素は下がる。window が 2 つだけなら両者は一致する。期待値と許容幅は
    イメージの中で 1 回測ってから決める。
  - 密な初期 window と高い threshold で、`pair` は全ての pair が threshold を超えて `converged`
    で終わり、`stage` は `max_windows` で終わる (合成のエネルギーを返す pool で
    `adaptive_schedule` を通す。変更前のコードで失敗することを 1 回確認)。
  - YAML の値が前計算に届く (`als_settings(cfg, leg)`: 判定、threshold、leg ごとの上限と長さ)。
  - 値の列で与えた初期 window がそのまま最初の iteration の window になり、ALS はそこから
    減らさない。candidate の上にない値、0 と 1 で始まり終わらない列は設定の段階で断られる。
  - leg ごとに違う設定 (判定、threshold、初期 window、mode) を書いた YAML で、それぞれの leg が
    自分の値を使い、`schedule.yaml` にその値が残る。1 つの値で書いた YAML は両方の leg に同じ値が
    渡る。知らない leg 名や片方だけの指定は設定の段階で断られる。
  - 既定値と文書の一致 (`test_documented_defaults_are_the_defaults_of_the_code`)。
  - 100 window の小さい系が replica exchange から解析まで通る。
- GPU: `scripts/in_docker.sh --gpus 0 python -m pytest tests/gpu -q` (前計算と pool を通る)。
- 評価 run: 各 run の `resolved_config.yaml` と `schedule.yaml` で、判定・threshold・初期 window が
  案のとおりであることを確認してから数値を報告する。同じコマンドをもう一度実行して、何も
  計算し直さないことを 1 run で確認する。

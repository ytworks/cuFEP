# 残っている記述の修正と lint (ruff) の導入
Status: In progress
Refs: PR #2 (merged as 99b59fe) and its independent review; this work is a new pull request, separate from PR #2

## Context

PR #2 (README 英日、開発ハーネス) は 99b59fe でマージ済み。その後に次の3点が残った。

1. **PR #2 の独立レビューの指摘**: P1 が 1 件、P2 が 10 件、P3 が数件。
   - README の記述が根拠を超えている (example の条件、`cufep plan` の説明、ドライバ要件、「判定はこれだけ」)。
   - 日本語版の誤訳がある (`NOT RELIABLE`)。
   - スクリプトの穴:
     - `in_docker.sh` が指定タグで黙ってビルドする。
     - `spikes/run.sh` が完全な互換になっていない。
     - 2つの検査スクリプトに誤判定がある。
   - P1 は、`fep-invariants.md` の「protocol.md が比較するプログラムに従う」が、商用製品に従うと言っているのと同じになっていること。
2. **既存の文書とコードのコメント**に、cuFEP の判断を商用 FEP 製品 (Schrödinger の FEP+・Desmond、Cresset の Flare) に従った・まねたとする記述が残っている。オーナーの規則では次のとおり。
   - 名前を書いてよいのは、精度の比較 (ベンチマークでの FEP+ の予測値との比較) と、出典の正しい記載だけ。
   - OpenFE、SOMD2、A3FE などのオープンソースの出典と、ruff などのツール名は問題にしない。
3. **lint の導入**: PR #2 で、別 PR に回すと決めたもの。

膜の向きを自動で決めるときの一致判定 (15°/5 Å で止まる、`prep/orient.py` の `consensus_orientation`) は cuFEP 独自の判定である。オーナーの回答により、今回は挙動を変えず「残っている独自の判定 (要判断)」として記録だけする。

## 方針

- **マージ済みの PR #2 とは別の、新しい PR で行う。** `main` (99b59fe) から新しいブランチ `fix/wording-and-lint` を作り、新しい draft PR を 1 本作る (PR #2 のブランチ `docs/oss-readme-harness` は使わない)。
- 手順ごとに commit・push し、push のたびに PR のレビューを読む。
- 計画は `docs/plans/2026-11-20-wording-and-lint.md` として commit する。
- コードの挙動は変えない。変えるのは次のものだけ。
  - 文書、コメント、docstring
  - lint のための書き換え (行の折り返し、import の整理、未使用の変数と import)
  - ハーネスのスクリプト
- 検証の基準は2つ。
  - 解析の数値はゴールデンテストで固定されていて、変わらない。
  - 単体テスト・GPU テストの件数が減らない。

## 手順

### 1. PR #2 のレビュー指摘を直す

文書:

- `fep-invariants.md`、`PROGRESS.md`、PR #2 の計画書の「確立したプログラムに従う」を、「文献と公開された方法 (例: OpenFE) に従い、各既定値の出典を configuration.md に書く」に改める。
- `fep-invariants.md` の判定表の「これで全部」を正確にする。
  - 文献の判定 (今の表)
  - 計算の規則: window の独立な標本が 4 未満なら自由エネルギーを出さない (`MIN_INDEPENDENT_SAMPLES`)。解の許容値 (`NEGLIGIBLE_FRACTION`、`RESOLUTION_KT`)。
  - 入力の制限: 正味電荷、未知の残基、明示的な水素のないリガンド。
  - 残っている独自の判定 (要判断): 膜の向きの一致 15°/5 Å。`PROGRESS.md` の未決事項にも書く。
- README の英日両版:
  - 日本語の `NOT RELIABLE` を「全 iteration で見たとき、または推定に使う標本で見たとき」に直す。
  - example の結果は、条件 3 点 (ALS 0.18、ポケットの REST2、production 5 ns) と「0.1 の以前のコミットで計算した」ことだけを書く。
  - `cufep plan` の説明を、`cli.py` の出力どおりにする (adaptive では開始の window)。
  - 「判定はこれだけ」を、「結果に付くメッセージ。解析の判定は文献の 2 つだけ」に改める。漏れていた注記の行 (`workflow.py` の、基準未満の overlap の部分) を足す。
  - ドライバ要件は、確認済みの事実 (RTX PRO 6000 Blackwell Server Edition、ドライバ 595.99.02、イメージの forward-compatibility パッケージ経由) と NVIDIA の互換表へのリンクにする。
  - P3:
    - `init` が書くのは最小の設定である。
    - 結果行の先頭には時刻が付く。
    - メッセージの表記を `LEG_FLAGS` の実際の文字列にそろえる。
    - お試し実行は `cufep benchmark run tyk2 --tier smoke` を案内し、「意味はない」という判定語を除く。
- `docker-execution.md`: `run_frozen.sh` は「同じ呼び方」ではないので、正しく書く。
- fmeta への言及にはリポジトリのリンクを付ける。
- PR #2 の計画書: テスト件数を 544 から 550 に直し、Status を Done (PR #2) にする。

スクリプト (それぞれ、レビューの再現入力を `tests/unit/test_harness.py` に固定する):

- `scripts/in_docker.sh`:
  - イメージが無いときにビルドしない。`--build` のときだけビルドし、無ければ止めて手順を示す。
  - `--gpus` に値が無ければエラーにする。
  - 新しいオプションを2つ足す。
    - `--docker-gpus VALUE`: docker の `--gpus` にそのまま渡す。
    - `--workdir DIR`: コンテナ内の作業ディレクトリ。
- `spikes/run.sh`: `--docker-gpus "${GPUS:-all}" --workdir /work` で呼び、以前の挙動と完全に同じにする。
- `scripts/check_readme_pair.py`:
  - `~~~` の囲みに対応する。行頭にインラインのバッククォートがあっても、囲みと誤認しない。
  - `#` は引用符の外にあるときだけコメントとみなす。
  - 次のリンクの書き方を読む: `<...>` で囲んだ先、`'title'` 付き、参照形式 `[x][r]` と `[r]: ...`、HTML の `href`。
  - `%20` などをデコードする。インラインコードの中はリンクとみなさない。
  - 見出しのアンカーを GitHub に合わせる: リンクを含む見出し、setext、1〜3 字下げ、`<a id>`、重複時の番号の付け方 (github-slugger と同じ)。
  - `####` 以下の見出しも比べる。
- `scripts/check_progress_size.py`:
  - コードの囲みの中は数えない。
  - Product 25 行、Current Status 80 行の上限も検査する。

### 2. 残っている「商用製品に従った・まねた」記述を除く

対象は、商用製品に従った・まねたとする記述をすべて (grep で洗い出し済み)。

| 対象 | 主な箇所 |
|---|---|
| `docs/protocol.md` | Terms の表、"as in FEP+ and Flare"、"the method of … Cresset Flare"、"Where cuFEP follows Cresset" の節 (論文 Midgley et al. 2025 との比較に改める)、REST2 の節の FEP+・Desmond の記述、"the remedy is the one of Flare"、"How cuFEP compares" の表 (FEP+ と Flare の列を除き、OpenFE との比較として残す) と、その Sources の商用製品の文書 |
| `docs/configuration.md` | production の行の "Schrödinger's absolute binding FEP+"、repeats の行の "FEP+ and Flare run a calculation once" (根拠を「1 化合物 1 回の計算で精度を出すことが要件」にする)、REST2 の行の FEP+ の温度、region の行、min_overlap の行の "warning level of Flare"、"(no other level)" の行 |
| `docs/plans/2026-11-18-abfe-engine.md` | 同じ趣旨の記述。ベンチマークの FEP+ との精度比較の数値は残す |
| コードのコメント・docstring | `mps.py` (Desmond)、`alchemy/restraints.py`、`config/schema.py`、`analysis/mbar.py`、`analysis/plots.py`、`sampling/states.py`、テストのコメント (`test_config.py`、`test_benchmarks.py`) |

残すもの:
- ベンチマークのコードと文書の FEP+ (精度の比較)
- Schrödinger ベンチマークの出典 (`registry.yaml`、`curate.py`、`examples/*/config.yaml`、`benchmarks/README.md`)
- OpenFE、SOMD2、A3FE、Boltz-ABFE の出典
- 設定値 `force_constants: openfe`

再発防止として `scripts/check_names.py` を作り、CI の `harness` ジョブに入れる (テスト付き)。

- `Flare|Cresset|Desmond` は、ライセンス表示 (`benchmarks/THIRD_PARTY/`、`docs/LICENSING.md`) と検査自身を除き、追跡ファイルのどこにも書かせない。
- `FEP\+|Schrödinger` は、精度比較と出典のファイル (ベンチマークのコード・データ・テスト・文書、README の該当節、`PROGRESS.md`、計画書、`examples/*/config.yaml`) にだけ許す。

### 3. lint の導入 (ruff 0.17.5、fmeta と同じ規則)

- `pyproject.toml` に設定を書く。
  - `[tool.ruff]`: `line-length = 120`、`target-version = "py312"`、`extend-exclude = ["runs", "spikes/out"]`
  - `[tool.ruff.lint]`: `select = ["E", "F", "W", "I", "B"]`、`ignore = ["E741", "B905"]`
- `scripts/lint.sh` を作る。ruff の公式イメージ `ghcr.io/astral-sh/ruff:0.17.5` (取得できることは確認済み) を、呼び出しユーザーでリポジトリに当てる。cuFEP のイメージは作り直さない。
  - CI の `harness` ジョブも `scripts/lint.sh` を呼ぶので、版は 1 か所で固定される。
  - `docker-execution.md`、`CLAUDE.md`、`CONTRIBUTING.md` に書き足す。
- 対象は追跡されている Python 全部: `src`、`tests`、`scripts`、`spikes`。現在、src 119 件、tests 185 件、spikes 194 件ある。直し方:
  - 自動修正 (`--fix`、安全なものだけ): I001、F401、E401、F541。これは独立の commit にする。
  - 手作業:
    - E501 (約 270 行): 行の折り返し。
    - E702 (約 100): 1 行 1 文にする。
    - B017 (13): `pytest.raises(Exception)` を、実際に投げられる例外の型にする。どの型かは走らせて確かめる。
    - B007、F841: `workflow.py` の未使用の `pre` と `tail` は、使われていないことを確かめて消す。
    - B023 (6): どれもループ内ですぐ使っている。既定引数で束縛する。
    - F821 (`sampling/equilibrate.py` の `del ctx, integ` による誤検出): 閉包が `del` の後に呼ばれないことを確かめたうえで、書き方を変える。
    - spikes の E402: `sys.path` を足した後の import なので `# noqa: E402`。
  - 調べた範囲では、F811 (テストの `signal` の再 import) にも B023 にも実害のあるバグは無かった。見つかった場合は直さずに報告する。

## 主なファイル

- 文書: `docs/protocol.md`、`docs/configuration.md`、`docs/plans/2026-11-18-abfe-engine.md`、`docs/plans/2026-11-20-oss-readme-and-harness.md`、`README.md`、`README.ja.md`、`CONTRIBUTING.md`、`CLAUDE.md`、`PROGRESS.md`、`.claude/rules/{fep-invariants,docker-execution}.md`
- スクリプト: `scripts/{in_docker.sh,check_readme_pair.py,check_progress_size.py}` を直し、`scripts/{check_names.py,lint.sh}` を新しく作る。`spikes/run.sh`、`.github/workflows/ci.yml`、`pyproject.toml`、`tests/unit/test_harness.py`
- lint の書き換え: `src/cufep/**`、`tests/**`、`spikes/*.py`

## 検証

1. `scripts/lint.sh` が終了状態 0。`scripts/check_names.py`、`check_readme_pair.py`、`check_progress_size.py` も終了状態 0。レビューの再現入力を固定したテストが合格する。
2. `scripts/in_docker.sh python -m pytest -q` が全件合格。件数は 553 + 新しいハーネスのテスト。golden (`test_the_numbers_of_the_analysis_cannot_change_without_a_new_version`) が変わらずに合格する。終了状態はパイプなしで読む。
3. `scripts/in_docker.sh --gpus 0 python -m pytest tests/gpu -q` が 17 件合格。
4. `spikes/run.sh` を偽の docker と本物で確かめる: `GPUS=2` が 2 枚の意味になること、作業ディレクトリが `/work` であること。
5. `IMAGE=cufep:missing spikes/run.sh` が、ビルドせずに止まること。
6. 独立なレビューエージェント (専用の新しい scratch ディレクトリだけを使う) に差分全体を攻撃させ、指摘を直す。
7. push、CI 緑、PR のレビューを読む。終わったら報告して、指示を待つ。

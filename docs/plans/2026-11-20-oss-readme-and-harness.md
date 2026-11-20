# OSS としての README (英語・日本語) と、fmeta と同じ開発ハーネス
Status: In progress
Refs: PR #1 (merged as 31a720e); fmeta (github.com/ytworks/fmeta) for the harness

## Context

オーナーの依頼 (2026-11-20、PR #1 のマージ後): 「README を OSS としてふさわしくきちんと書く。日本語版と英語版の両方を作る。開発のハーネスを fmeta と同様に作る。PR を作る」。

現状:
- `README.md` は英語のみ。「(owner, 2026-11-20)」のような内部の書き方が残り、状態の説明は `docs/PLAN.md` (日本語、改訂ログ) を指している。バッジ、目次、要件、ドキュメント一覧、引用文献、コントリビュートの案内がない。
- 開発ハーネスがない。プロジェクトの規則 (公知の判定基準だけ、確立した FEP ソフトウェアに従う、既定は 1 回、cuPotential の速度を載せない等) は Claude の非公開メモリにしかなく、別のエージェントや人には見えない。
- fmeta (`ytworks/fmeta`) のハーネス: `CLAUDE.md` (セッション開始時に `PROGRESS.md` を読む、規則の `@import`、テスト方針、配置、開発コマンド)、`AGENTS.md` (Codex 向けに CLAUDE.md を指す)、`.claude/rules/` (docker-execution、progress-log、numerics-invariants)、`.claude/settings.json` の SessionStart hook (`scripts/check_progress_size.py`)、3 節構成で行数上限付きの `PROGRESS.md` とアーカイブ手順 (`docs/progress/`)、`docs/plans/` の計画書、`scripts/in_docker.sh`、CI での検査、`tests/test_harness.py`。

決定済み: lint (ruff) はこの PR に入れない (オーナー回答)。`src/` のコードは一切変えない。

## 方針

- 新しいブランチ `docs/oss-readme-harness` を `main` (31a720e) から作り、draft PR を 1 本作る。手順ごとに commit・push し、push ごとに PR のレビューを読む。手順の前後に日本語で報告する。
- fmeta の仕組みはそのまま移し、中身は cuFEP のものにする。fmeta 固有で cuFEP に対応物がないもの (conda lock の照合、`bench/results` の証跡索引、D-ID 付きの design 文書) は移さず、理由を計画書に書く。
- README の記述はすべてリポジトリの文書とコードで確かめる。数値は出典 (文書の節) のあるものだけ、判定語なし。cuPotential の速度・所要時間は一切書かない (NVIDIA ライセンス)。

### オーナーの指示 (2026-11-20、実装中)

「他社の名前やプロダクト名は一切書かないで」、続けて「ただし精度を比較する時は例外で ok / 意思決定や真似た事実などを書いてほしくない趣旨 / 正しく出典を書くのは ok / ruff とかそういうのはいい」、問題は商用の FEP 製品の会社であること。回答と合わせて次のとおり。

- この PR で書く文書 (README 両版、`CONTRIBUTING.md`、`CLAUDE.md`、`.claude/rules/`、`PROGRESS.md`、この計画書) に、cuFEP の判断が商用の FEP 製品やその会社に従った・まねた・それを見て決めた、という記述を書かない。他の FEP ソフトウェアとの機能の比較の節は削除する。
- 名前を書いてよいのは、精度の比較 (ベンチマークのレポートで FEP+ の公開予測値と比べる) と、出典の正しい記載 (Schrödinger の公開ベンチマーク、ライセンスの表示) だけ。依存物 (OpenMM とイメージ内のパッケージ、NVIDIA の GPU・CUDA・MPS・cuPotential、Docker) と開発ツール (ruff など) の名前は問題にしない。
- 既存の文書 (`docs/protocol.md`、`docs/configuration.md`、`examples/`・`benchmarks/` の README、これまでの計画書) とコードは今回は変えない。
- この規則は `.claude/rules/fep-invariants.md` と `CONTRIBUTING.md` に書く。
- その後のオーナーの指示「やりすぎ / 趣旨を理解して戻すべきところは戻せ」により、OpenFE (オープンソース。規則の出典) と開発ツール・エージェントの名前 (Claude Code、Codex) は戻した。消したままにするのは、cuFEP の判断を商用の FEP 製品に従った・まねたとする記述と、それらを並べた機能の比較表だけ。

## 手順

### 0. ブランチと計画書
- `git switch -c docs/oss-readme-harness main`
- この計画を `docs/plans/2026-11-20-oss-readme-and-harness.md` として commit (Status 行付き)。
- `docs/PLAN.md` を `git mv` で `docs/plans/2026-11-18-abfe-engine.md` へ移す (中身は不変、先頭に `Status: Done (PR #1)` の 2 行だけ追加)。参照箇所 (README、protocol.md 等) を grep して直す。

### 1. 開発ハーネス (fmeta との対応)

| fmeta | cuFEP で作るもの |
|---|---|
| `CLAUDE.md` | `CLAUDE.md`: cuFEP の概要、Session start (`PROGRESS.md` を最初に読む)、3 つの規則の `@import`、テスト方針 (fmeta の 7 項目を cuFEP に合わせる: CPU 単体は `tests/unit`、`-m gpu`/`cupotential`/`physical` の印、修正前に落ちる回帰テスト、時間で判定しない)、配置 (`src/cufep/` の各パッケージ、`docs/protocol.md`/`configuration.md`、`examples/`、`benchmarks/`、`spikes/`、`docs/plans/`)、開発コマンド (パイプで終了状態を隠さない、を含む) |
| `AGENTS.md` | 同じ 1 行: `Read and follow [CLAUDE.md](CLAUDE.md).` |
| `.claude/rules/docker-execution.md` | cuFEP 版: すべて公式イメージ (`docker/Dockerfile`) の中で `scripts/in_docker.sh` 経由で走らせる。例外は CI と SessionStart hook だけ。Docker のないセッションは「ここでは未検証」と書く。NVIDIA のソフトのライセンス、`WITH_CUPOTENTIAL=0` の stock 版、`--shm-size`、MPS は cuFEP 自身が管理すること |
| `.claude/rules/progress-log.md` | ほぼ同文。D-ID の代わりに「`Current Status` は `docs/protocol.md`・`docs/configuration.md` と食い違わない」。英語で書く |
| `.claude/rules/numerics-invariants.md` | `.claude/rules/fep-invariants.md`: cuFEP の規則を公開の形で: 判定は公知の基準だけ (overlap ≥ 0.03 Klimovich 2015、無相関サンプル ≥ 50 Mey 2020、MBAR の連結、箱の高さ 2×/3× cutoff、ALS 閾値 0.18、OpenFE の Boresch アンカー規則)。他は判定なしの数値。既定値と用語は確立した FEP ソフトウェアと文献に従い出典を protocol.md に書く。1 回の計算が既定 (`sampling.repeats: 1`)。全設定は YAML で既定値付き。自動は名指しのもの (Boresch restraints、adaptive lambda windows、GPU 割当) だけ、膜は設定、production の自動延長なし、退避は必ずログ。cuPotential/ALS/REST2 を「有効だが働かない」設定にしない。cuPotential の速度は追跡ファイル・commit・GitHub に書かない。基準を緩めて通さない。指摘はその類全体を共有関数で直す。各規則に検査しているテスト名を付ける (実装時に grep で確認したものだけ) |
| `.claude/settings.json` | 同じ SessionStart hook |
| `PROGRESS.md` + `scripts/check_progress_size.py` | 同じ 3 節 (Product / Current Status / Change Log)、200 行・10 項目の上限。Product は承認済み計画の要約、Current Status は 31a720e 時点の事実 (テスト件数、examples の最新結果は PLAN.md/protocol.md にある数値のみ、未実施の benchmark 等)、Change Log は PR #1 のマージとハーネス導入の 2 項目 |
| `docs/progress/archiving.md`、`archive/change-log.md` | 同じ (マーカー `ARCHIVE-INSERT-POINT` 付きの空のアーカイブ) |
| `docs/plans/README.md` | 同じ書式 (Status / Refs / Context / Approach / Invariants / Verification) |
| `scripts/in_docker.sh` | cuFEP 版の汎用ラッパー: `--gpus`、`--detach`、`--name`、`--mount`、`--image` (既定 `$CUFEP_IMAGE` か `cufep:dev`)、`--build`。`spikes/run.sh` が今設定している環境 (`PYTHONPATH=/work/src`、`CUFEP_DATA=/work`、`CUDA_MPS_PIPE_DIRECTORY`、`OPENMM_CPU_THREADS`、`--shm-size`、`--ulimit core=0`、呼び出しユーザー) をこちらに一本化し、`spikes/run.sh` は `scripts/in_docker.sh --gpus "${GPUS:-all}" python "$@"` を呼ぶだけの互換シムにする (既存の呼び方はそのまま動く) |
| `tests/test_harness.py` | `tests/unit/test_harness.py`: `problems_in` の節・行数・項目数の検査と、README 対の検査 (標準ライブラリのみ) |
| CI の検査 | `.github/workflows/ci.yml` に軽いジョブ `harness` (setup-python、`check_progress_size.py` と `check_readme_pair.py`) を追加。既存の `unit` ジョブは変えない |

移さないもの: conda lock と照合 (cuFEP は lock を持たない)、`bench/results` と `verify_bench_results.py` (cuFEP は GPU 検証を `tests/gpu` で行う)、D-ID 付き design 文書 (cuFEP の仕様は `docs/protocol.md`)、ruff (オーナー回答で別 PR)、image 公開ワークフロー (NVIDIA ライセンスの確認待ちという承認済みの前提)。

`.gitignore` に `.claude/worktrees/` と `.claude/settings.local.json` を足す。

### 2. README (英語 `README.md`、日本語 `README.ja.md`)

両方の先頭に言語の切り替え (`English | 日本語`)。同じ節構成:

1. 題名、1 段落の説明、バッジ (CI、Apache-2.0、Python 3.12)
2. 状態 (v0.1 開発版、中性リガンドのみ等) と目次
3. Features
4. How it works (準備 → pre-equilibration → Boresch restraint の選択 → adaptive lambda windows と HREX/REST2 → MBAR と検査; 各段の protocol.md の節へのリンク)
5. Requirements (Linux、Docker + NVIDIA Container Toolkit、ドライバ条件)
6. Installation (イメージの build、`WITH_CUPOTENTIAL=0` の stock 版、LICENSING へのリンク)
7. Quick start (`check-env` → `init` → `plan` → `run`)
8. Outputs (出力ファイルの表)
9. Reading the results (メッセージの表を、内部の言い回しを除いて残す。leg の再実行例)
10. Configuration (最小の YAML、よく変える設定、configuration.md へ)
11. Examples (TYK2、A2A、実験値)
12. Benchmarks (`cufep benchmark`、Schrödinger の公開ベンチマーク。実験値と FEP+ の予測値との精度比較)
13. (削除: 他のソフトウェアとの比較の節。下の「オーナーの指示」)
14. Limitations (v0.1)
15. Documentation (protocol、configuration、LICENSING、M0_REPORT、plans、PROGRESS)
16. Contributing / Development (`CONTRIBUTING.md`、`scripts/in_docker.sh`、ハーネス)
17. References (Boresch 2003、REST2 Wang 2011、MBAR Shirts & Chodera 2008、ALS Midgley 2025、Klimovich 2015、Mey 2020、OpenMM、ベンチマークの論文 Ross 2023) — 書誌は protocol.md と論文で確認したものだけ
18. License と第三者ソフト

日本語版でも分野の英語の用語はそのまま使う (lambda window、replica、complex leg / solvent leg、repeat、overlap matrix など)。コードブロックのコマンドは両版で同一。

`CONTRIBUTING.md` (英語): 開発環境 (イメージ)、テストの走らせ方 (CPU、GPU)、PR とレビューの進め方、ハーネスの各ファイルの役割、cuPotential の速度を載せない規則。

`scripts/check_readme_pair.py` (標準ライブラリ): `README.md` と `README.ja.md` の `##`/`###` 見出しの数と並び、相対リンクの集合、コードブロック内のコマンド行が一致することを検査し、相対リンクの先が存在することも確かめる。

### 3. 確認と PR
- 下の検証をすべて通し、独立なレビューエージェント (自分専用の新しい scratch ディレクトリ、それ以外は消さない) に README 両版の記述をコードと文書に照らして攻撃させ、指摘を直す。
- push、draft PR 作成 (本文はこの計画の要約、検証結果、移さなかったものと理由)。PR のレビューと CI を読む。

## 主に変える・作るファイル
- 新規: `CLAUDE.md`、`AGENTS.md`、`.claude/settings.json`、`.claude/rules/{docker-execution,progress-log,fep-invariants}.md`、`PROGRESS.md`、`docs/progress/{archiving.md,archive/change-log.md}`、`docs/plans/README.md`、`docs/plans/2026-11-20-oss-readme-and-harness.md`、`scripts/{in_docker.sh,check_progress_size.py,check_readme_pair.py}`、`tests/unit/test_harness.py`、`README.ja.md`、`CONTRIBUTING.md`
- 変更: `README.md` (書き直し)、`spikes/run.sh` (シム化)、`.github/workflows/ci.yml` (ジョブ追加)、`.gitignore`、`docs/PLAN.md` → `docs/plans/2026-11-18-abfe-engine.md` とその参照
- 変えない: `src/` 全体、`docker/`、既存テスト

## 検証
1. `scripts/in_docker.sh python -m pytest tests/unit -q` が全件合格 (現在 544 件 + 新しいハーネスのテスト)。終了状態はパイプなしで読む。
2. `spikes/run.sh -m pytest tests/unit -q` もシム経由で同じ結果 (互換性)。`scripts/in_docker.sh --gpus 0 python -m pytest tests/gpu -q` を 1 回 (ラッパーの GPU 経路; 17 件)。
3. `python3 scripts/check_progress_size.py` と `python3 scripts/check_readme_pair.py` が終了状態 0 (hook と CI と同じ呼び方)。わざと 11 項目にした PROGRESS.md で失敗することをテストで確認。
4. README の全コマンドとオプションを `scripts/in_docker.sh cufep <cmd> --help` で、YAML のキーと既定値を `cufep init --full` の出力で照合。
5. 全 Markdown の相対リンクが存在することを `check_readme_pair.py` と grep で確認。cuPotential の速度・所要時間の語 (min、hours、ns/day、速度) が README 両版・CONTRIBUTING・PROGRESS にないことを grep。
6. push 後、GitHub 上で README の描画とバッジを確認。CI (`unit`、`harness`) が緑。PR のレビューを読み、指摘があれば直す。

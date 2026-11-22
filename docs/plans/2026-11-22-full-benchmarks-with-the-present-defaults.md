# 現在の既定値でベンチマーク full を 1 回実行し、wiki に記載する
Status: In progress (手順 1〜2 済み: イメージ e77a6c5、draft pull request; 手順 3: 計算中)
Refs: this pull request; `benchmarks/results/`; `docs/wiki/Accuracy.md`, `docs/wiki/Accuracy-ja.md`; `docs/plans/2026-11-20-full-benchmarks-and-accuracy-wiki.md`

## Context

オーナーの依頼 (2026-11-22): 新しい既定値でベンチマークを 1 回全て実行して wiki に記載する。テストと
レビューは不要。draft pull request を作ったらすぐに計算を始める。

事実:
- 記録済みの結果 (`benchmarks/results/`、`docs/wiki/Accuracy.md`) はコミット 3b45bfe の計算で、既定値が
  今と 2 点違う。restraint の選び方が `restraints.selection: auto` (今は `interactions`)、complex leg の
  adaptive lambda scheduling の開始が 3、5、9 window で stage の overlap 行列の 0.18 (今は 30 window で
  pair の overlap の 0.30)。今の既定値では両セットの smoke tier しか実行していない。
- example の `config.yaml` もベンチマークの設定も既定値に従う (`src/cufep/benchmarks/registry.yaml`)。
  ファイルを変えずに今の既定値で動く。この計画はコードを変えない。
- GitHub Wiki にはページがまだない (2026-11-22 に確認)。wiki の原稿は `docs/wiki/`。

## Approach

1. `main` のコミット e77a6c5 (PR #7 のマージ) から凍結イメージ `cufep:e77a6c5` を作る。
2. draft pull request (この計画と `PROGRESS.md`)。
3. TYK2 full (16 リガンド)、続けて A2A full (17 化合物) を、それぞれ 8 GPU で 1 回
   (`spikes/run_frozen.sh benchmark run <set> --tier full`)。1 リガンド 1 計算。subset tier は full に
   含まれる同じ設定の 5 リガンドなので別には実行しない。
4. 結果を `benchmarks/results/<set>/` に記録する (計算が書いたレポート、表、図、`summary.csv`、
   `summary.json`、`resolved_config.yaml`)。前の計算の記録はコミット cda6965 の木に残る。実行時間は
   書かない (`docs/LICENSING.md`)。
5. `docs/wiki/Accuracy.md` と `Accuracy-ja.md` を今回の計算で書き直す (同じ構成、同じ数値、図は今回の
   結果だけ)。前の計算 (3b45bfe、前の既定値) の統計量と並べる節を 1 つ置き、違う設定を明記する。
6. 前の計算の数値を今の状態として書いている場所を更新する: README (英語・日本語)、`PROGRESS.md`、
   `docs/protocol.md`、`docs/configuration.md`、`benchmarks/results/README.md`。日付のある履歴は
   書き換えない。
7. マージ後、GitHub Wiki にページができていれば公開する (最初のページはオーナーが Web UI で作る)。

## Invariants

- 判定は文献の基準だけ (overlap 0.03、独立サンプル 50)。`NOT RELIABLE` や独立サンプル不足の結果も
  除外せず、1 回の計算のまま記録して実験値と FEP+ と比べる。再実行しない。cuFEP が基準で止めた
  リガンドは結果なしとして理由を記録する。判定語 (converged など) を書かない。
- 1 リガンド 1 計算。自動の延長・再実行はしない。中断した計算は同じコマンドで再開する (終わった leg は
  計算し直されない)。
- FEP+ と Schrödinger の名前は精度の比較と出典にだけ使う。
- cuPotential の速度と実行時間は追跡ファイル、コミット、GitHub に書かない。

## Verification

- オーナーの指示で、単体テスト、GPU テスト、レビューは実行しない (コードを変えない)。CI は push で動く。
- 計算: 両セットの終了コード、結果のあるリガンドの数、設定は計算のファイル (`resolved_config.yaml`、
  各 leg の `schedule.yaml`) で確かめる。
- 記録: `cufep benchmark report <set> <run directory>` の数値が記録と wiki に一致すること。記録した
  ファイルに実行時間がないこと。`scripts/check_progress_size.py`、`scripts/check_readme_pair.py`、
  `scripts/check_names.py`。

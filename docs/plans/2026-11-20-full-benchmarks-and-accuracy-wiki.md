# ベンチマーク full の実行と、精度を説明する wiki (英語・日本語)
Status: In progress (steps 1-6 done: benchmarks with the example settings, the report, the frozen image 3b45bfe, both full runs, the record in benchmarks/results, the wiki pages; step 7 left: review, merge and publication on the GitHub Wiki)
Refs: this pull request; `docs/configuration.md` (`sampling.production_ns`); `benchmarks/README.md`

## Context

オーナーの依頼 (2026-11-20): ベンチマークを example と同じ設定で動くようにし、ベンチマークを全て
実行して結果をまとめて記録し、実験値と FEP+ の予測値との比較を必須とした cuFEP の精度を説明する
wiki を英語と日本語で作る。

調べて分かった事実:
- `cufep benchmark run <set> --tier full` は今でも example と同じ入力と同じ既定値で動く。タンパク質の
  PDB はバイト単位で一致し、example のリガンド (TYK2 ejm_31、A2A 4e) はセットの SDF のレコードと
  座標が一致する。違いはリガンドの数だけ。
- ただし同じであることはコードで保証されていない。`src/cufep/benchmarks/registry.yaml` が膜の有無を
  別に持ち、`benchmark_config` が設定を独自に組み立てている。
- README の example の数値 (TYK2 -8.94 ± 0.77、A2A -11.95 ± 0.73 kcal/mol) は production 5 ns/窓 で
  得た。既定値は 10 ns。
- full セット (TYK2 16 リガンド、A2A 17 化合物) はまだ実行していない。
- GitHub Wiki は有効だがページがない。

オーナーの決定:
- production は 5 ns/窓 (README の example の数値の設定)。example の `config.yaml` に書き、
  ベンチマークはそれに従う。既定値の 10 ns は変えない。
- wiki の原稿はこのリポジトリの `docs/wiki/` に置いて pull request でレビューし、マージ後に GitHub
  Wiki に公開する。
- フラグの付いたリガンドは 1 回の計算のまま記録する (再実行しない)。
- NOT RELIABLE のリガンドも除外せず、実験値と FEP+ と同じように比較する。
- この計画の承認が、手順 1〜3 がきれいになった後の計算開始の許可 (開始と終了を報告する)。

## Approach

1. ベンチマーク = example の設定を構造で保証する。`registry.yaml` の各セットは example の
   `config.yaml` を指し、`benchmark_config` はその設定を読んで、入力ファイル (タンパク質とセットの
   リガンド) と出力先だけを置き換える。subset はリガンド名、smoke はその上に短縮したサンプリング。
   両 example の `config.yaml` に `sampling.production_ns: 5.0`。テスト: full / subset の設定が入力と
   出力を除いて example の設定と一致し、セットのタンパク質が example のものとバイト単位で一致し、
   example のリガンドのレコードがセットにある。
2. レポートの比較: 統計の主表は結果のある全リガンド (NOT RELIABLE を含む)。その下に NOT RELIABLE
   のリガンドとそれ以外の内訳 (cuFEP と FEP+ を同じリガンドで: 件数、平均符号誤差、RMSE、MUE、
   平均誤差を除いた RMSE)。リガンドごとの表に cuFEP − 実験、FEP+ − 実験、0.03 を下回った leg と
   窓の組。図では NOT RELIABLE の点を別の記号で描く。
3. 凍結イメージ `cufep:<commit>`、`check-env`、GPU テスト。
4. TYK2 full と A2A full を 1 本ずつ 8 GPU で (`spikes/run_frozen.sh benchmark run <set> --tier full`)。
5. 結果を `benchmarks/results/<set>/` に記録する (レポート、表、図、`summary.csv`、
   `resolved_config.yaml`、実行条件の README)。実行時間は書かない (`docs/LICENSING.md`)。
   README の Examples と Benchmarks、`PROGRESS.md` を更新する。
6. wiki: `docs/wiki/Home.md`、`Accuracy.md` (英語)、`Accuracy-ja.md` (日本語)。両言語で同じ構成、
   同じ数値。`scripts/check_names.py` が `docs/wiki/` を精度比較のファイルとして認め、
   `scripts/check_readme_pair.py` が wiki の英日ペアと表の数値を照合する。
   オーナーの追加の指示 (2026-11-21): 図表は必須、図には今回の計算の結果だけ、相関係数も入れる、
   別の ABFE 手法 Felis (ByteDance Seed、arXiv:2603.22274 版 3) とも比較する。Felis はリガンド
   ごとの値を公開していないので、標的ごとに論文の定義の統計量 (R²、ρ、τ、オフセット b、補正後
   RMSE と MAE、順序正解率) で比べる。この定義を公開された FEP+ の予測値に当てはめると論文の
   FEP+ の行と一致することを確かめた。
7. 独立レビュー、push、CI。マージ後に GitHub Wiki へ公開する (最初のページはオーナーが Web UI で
   作る)。

## Invariants

- 判定は文献の基準だけ (overlap 0.03、独立サンプル 50)。NOT RELIABLE は除外の理由にしない。
  判定語 (converged など) を書かない。
- FEP+ と Schrödinger の名前は精度の比較と出典にだけ使う。cuFEP の設定や判断がそれに従ったとは
  書かない。
- cuPotential の速度と実行時間は追跡ファイル、コミット、GitHub に書かない。
- 1 リガンド 1 計算。自動の延長・再実行はしない。

## Verification

- CPU: `scripts/in_docker.sh python -m pytest -q`、`scripts/lint.sh`、harness の 3 スクリプト。
- GPU: 凍結イメージで `cufep check-env` と `tests/gpu`。
- 計算: 両セットの full run。記録した `benchmark_report.*` を `cufep benchmark report` で作り直して
  wiki と一致すること。記録したファイルに時間が無いこと。

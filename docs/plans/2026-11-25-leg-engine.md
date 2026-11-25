# レグのエンジンを分ける(ABFE と RBFE を 1 つのエンジンの上の 2 つのプロトコルにする)
Status: Implemented, in review (branch `refactor/leg-engine`, draft PR #18 on `feat/rbfe`)
Refs: `docs/plans/2026-11-25-rbfe.md`; PR #17

## Context
- オーナーの問い(2026-11-25): 「無理に共通化してない? ソフトウェアとして正しいアーキテクチャか」。今の `cufep rbfe` は
  ABFE の入口に合わせて押し込んでいる:
  1. RBFE の設定から ABFE の `Config` を作って共通の関数に渡し(`RbfeConfig.shared()`)、前平衡化の動作を ABFE の拘束の
     設定(`restraints.selection: interactions`)で操っている;
  2. レグの結果を ABFE の `leg_result_from_samples` で作ってからキーを付け替え、拘束の項目を消し、メッセージを文字列置換;
  3. `HybridLegJob` が ABFE の `LegJob` を継承し、`ligand` にエッジ名、`lig_dir` にエッジのディレクトリを入れている;
  4. スケジューラは `getattr(job, "runner")`、状態は辞書のキー(`lambda_hybrid`)で振り分け、共通の `sampling/states.py` が
     RBFE の stage 名とパラメータ名を知っている;
  5. RBFE のレグ(`rbfe/leg.py`)が ABFE の `run_leg` のバックエンド選択、fallback、`LegSpec` の組み立てを写している。
- オーナーの指示: この計画で進める。`feat/rbfe`(PR #17)をベースに別の PR にする。

## Approach
- `src/cufep/engine/`(新規、ABFE と RBFE が使う中核):
  - `settings.py`: レグの設定(熱力学、動力学、バックエンド、前平衡化、窓、サンプリング、解析、資源)。前平衡化は
    「溶質を固定するか」「自由な MD の ns」「二重層の平衡化の ns」を明示する。YAML の schema から作る。
  - `protocol.py`: 状態の型、stage、状態エネルギーの評価、レグの両端、窓の推奨値をまとめたプロトコル。ABFE と hybrid の
    2 つ。プロトコル名を `LegSpec` の設定と `schedule.yaml` に記録し、読み込みはその名前で振り分ける(辞書のキーで
    判別しない。名前の無い古いファイルは ABFE として読む)。
  - `leg.py`: レグの実行の中核。前平衡化 → 窓(ALS、固定、与えられた窓、窓の開始) → HREX(`sample_leg`) → バックエンドの
    fallback → 再開 → レグの解析。返すのは経路の ΔG、誤差、検査、窓。ABFE の語を使わない。
  - `jobs.py`: 汎用のジョブ(名前、レグ、repeat、ディレクトリ、設定、実行関数)。スケジューラはこれだけを知る。
- ABFE: `workflow.run_leg` は系の読み込み、拘束の選択(前平衡化の後の hook)、alchemical の System、拘束の解放、
  `result.json` の作成だけになる。`result.json`、`LegSpec.conditions()`、ディレクトリの形は今のまま。
- RBFE: 自分の設定から直接エンジンの設定を作る。`shared()`、結果の付け替え、`HybridLegJob` の継承、文字列置換は削除。
- 設定: 2 つの YAML に共通の節を `CommonConfig`(`config/schema.py`、節の型と同じモジュールにして循環 import を避ける)に
  置き、ABFE の `Config` と `RbfeConfig` がそれを継承する。前平衡化で溶質を固定するかは各設定が `holds_solute(leg)` で
  答える。YAML の形はどちらも変えない。

## Invariants
- ABFE の数値、既定値、`ANALYSIS_VERSION`(10)、YAML の形、`result.json` のキー、`LegSpec.conditions()`、既存の run
  ディレクトリの再開と解析は変えない。RBFE の既定値と結果の形(`edges.csv`、`ligands.csv`、`summary.json`)も変えない。
- 新しい判定は足さない。

## 手順(1 つずつコミットして push)
1. ABFE の今の振る舞いを固定するテストを先に足す: 小さな系の alchemical System の XML の checksum、`LegSpec.conditions()`、
   `result.json` のキー、前平衡化の各部の長さ。リファクタリングの後も同じであることを確かめる。
2. `engine/settings.py` と `engine/protocol.py`。ABFE と RBFE の設定から作る。
3. `run_leg` の汎用の部分を `engine/leg.py` に移し、ABFE は hook と後処理だけにする。
4. RBFE のレグをエンジンの上に載せ直し、流用していた箇所を削除する。
5. スケジューラを汎用のジョブにする。
6. 文書(`docs/protocol.md`、`docs/rbfe.md`)、PROGRESS、この計画の記録。

## Verification
- CPU: 単体テスト全体、手順 1 の固定テスト、golden、RBFE のエッジの一連の流れ(each、solvent、fixed)。
- GPU: GPU テスト全体。
- 実行: ABFE は既存の run(`runs/benchmark_tyk2_full_e77a6c5` の 1 化合物)を `cufep analyze` で読んで結果が同じこと。
  RBFE は TYK2 の example を 1 回流し、エッジの数値が前回(image f6ac3b4)の誤差の範囲にあること。

## 記録

### 2026-11-25 手順 1〜6
- 手順 1: `spikes/m38`(ABFE の 2 レグ、Reference platform、固定 seed)。リファクタリング前のコードで 2 回流して同じファイル
  (CPU platform は deterministic forces 無しでは 2 回目で前平衡化の後の System が違った)。
- 手順 2〜5 は計画の通り。違えた点: `CommonConfig` は `config/common.py` ではなく `config/schema.py`(節の型と同じ
  モジュール、循環 import を避ける)。ABFE の窓の検査(`stored_windows`、`check_windows_settings`)は設定の代わりに
  窓の設定の辞書を受け、相対レグも同じ検査を通る。手順 3〜5 は互いに同じ行に触るので 1 つのコミット(250ceb1)。
  レグの補助関数(前平衡化、HREX、fallback の判定など)は今は `workflow.py` に残り、エンジンがそれを呼ぶ。
- ABFE が変わらないこと: m38 の指紋が手順 2 の後(89477d2)、手順 3〜5 の後(250ceb1)ともリファクタリング前と
  バイト単位で同じ。保存済みの ABFE の run(TYK2 ejm_31 の 2 レグ、image e77a6c5)の解析の結果の辞書が前後で同じ。
- 見つかったこと: `feat/rbfe` のエッジのテストは CPU platform(deterministic forces 無し)と乱数で置く水素のため、
  全体の中では 1 本目のレグの窓が繋がらず失敗し、単独では通った。deterministic forces と seed で 2 回同じ数値に
  (`feat/rbfe` の 5b56034)。
- 単体テスト全体(5cc23eb): 778 passed。GPU テスト(GPU 0): 61 passed、4 failed(hybrid のレグの HREX のテストが
  `LegSpec` にプロトコルを書いておらず、worker が状態を ABFE として読んだ: テストを直した)。
- TYK2 の相対の example を image 5cc23eb で(GPU 1〜7、全レグ プラグイン)。全レグが終わった後、run 全体の timing の
  記録が各レグの ligand を読み `KeyError: 'ligand'` で止まった(エッジのレグには無い): ジョブの名前で書くように直し、
  回帰テスト(直す前のコードで KeyError)。数値(`cufep rbfe analyze`、kcal/mol、前回 f6ac3b4): ejm_31→ejm_46
  -1.75 ± 0.27(-1.86 ± 0.29)、ejm_31→ejm_49 +1.09 ± 0.44(+0.56 ± 0.32)、ejm_46→ejm_49 +2.86 ± 0.90(+2.39 ± 0.32、
  NOT RELIABLE: complex レグの推定のサンプルの overlap、窓 9、前回は 10)。3 本とも差は二つの誤差の合成の範囲内。
  cycle closure +0.03 ± 1.04。dG -9.31 ± 0.17、-11.06 ± 0.20、-8.22 ± 0.27(実験 -9.54、-11.31、-7.75)。

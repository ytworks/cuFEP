# 非拘束 MD の相互作用解析 (ProLIF) から Boresch 拘束を決めるモード
Status: In progress (steps 1-13 done: the plan, the environment, the plain MD of a leg, the trajectories, the rule, ProLIF, the selection in the leg, the outputs, the GPU test, the default, the review of the whole diff and the fixes of its findings; left: step 14, one run of each example)
Refs: this pull request; `docs/protocol.md` ("Steps of one leg"); `docs/configuration.md` (`restraints`)

## Context

オーナーの依頼 (2026-11-21): Felis のように MD を実行して拘束条件を決める機能を作る。MD の解析には
ProLIF を使い、定量的に解析して拘束を決める材料にする。入力は今までと同じ PDB と SDF で、ABFE の
実行まで自動。モードは YAML で選び、現在の方式も選べる。example で正しく動くところまで。
この MD も YAML の設定どおりに動かし、cuPotential が設定されていれば他の部分と同じように使う。

オーナーの決定 (2026-11-21):
- 新しいモードを既定値にする (`restraints.selection: interactions`)。現在の方式は `auto` で選べる。
- 論文に書かれていない部分は Felis のソースコードも読んで決める。
- example での確認は TYK2 と A2A の両方、各 1 回。
- 拘束用の MD は YAML の設定とバックエンド (cuPotential) に従う。ドラフト pull request を作る。

調べて分かった事実 (cuFEP):
- 現在: `restraints.selection: auto | manual`。complex leg の pre-equilibration (非拘束 NPT 5 ns、
  20 ps ごと、溶質原子のみ、`equil.npz`) のフレームから `find_boresch_restraint`
  (`src/cufep/alchemy/restraints.py`) が芳香環と安定な主鎖を選ぶ (OpenFE の方式)。相互作用は見ない。
  参照値は最後のフレーム、力の定数は 10 kcal/mol/A^2 と 80 kcal/mol/rad^2 (すべて 0.5 k x^2)。
- 平の MD の関数は既にある (`sampling/equilibrate.py` の `_production`)。pre-equilibration の本番部分は
  `backend.name` のバックエンドで走り、動かせない・失敗したときは stock OpenMM でやり直して記録する
  (`_run_once`)。ただしこの切り替えは `backend.fallback: none` を見ていない (アルケミカル計算の
  `_choose_backend` は見ている)。
- 平の MD のジョブは GPU 1 枚。`JobControl.request_devices(n, work, pinned=True)` で複数の GPU を
  借りられる (adaptive lambda scheduling の事前計算が使っている)。
- `run_leg` の拘束の段は `manual` か「それ以外」の 2 分岐で、3 つ目の値は黙って現在の探索に入る。
- `prolif` と `MDAnalysis` は `docker/environment.yml` に無い。`mdtraj` と `rdkit` はある。
- ベンチマークは example の `config.yaml` を読むので、既定値を変えると両ベンチマークの設定も変わる。
- `restraint_made_with` は `restraints` 節全体を記録し、記録に無いキーは「違い」として拒否される。

## 出典: Felis の方法 (論文とソース)

Ding, Liu, Wang, Xia, Wang, Yan, "Development and large-scale benchmarks of a protein-ligand absolute
binding free energy toolkit", arXiv:2603.22274 v3 (2.1 節、2.4 節)。ソース: github.com/ByteDance-Seed/felis、
commit d6dff91 (2026-10-19、v0.2.0): `felis/protocols/boresch/`、`felis/external/prolif_tools.py`、
`felis/protocols/abfe_v2/stage_boresch.py`、`submodule/prolif.patch`。ライセンスは CC BY-NC 4.0 なので
コードは写さない。規則と数値を出典つきで採り、実装は独自に書く。

流れ: 箱を作る、最小化、拘束用 NPT、アンカーと参照値、溶媒 leg、complex leg、解析。complex leg は
最小化した入力構造から始まる (NPT の軌跡は拘束を決めるためだけに使う)。

1. MD: 同じ温度 298.15 K・同じ状態 (完全に相互作用、拘束なし) の 4 レプリカを NPT、4 fs、各 300 フレーム
   x 10 ps = 3 ns。各レプリカの最初の 100 フレーム (1 ns) を捨てて連結 (800 フレーム)。状態が同じなので
   独立な 4 本と同じ。別の選択肢 `long_npt` は 1 本 12 ns。
2. ProLIF (2.0.3 とパッチ): タンパク質全体、`count=True`、`vicinity_cutoff=6.0` A、幾何は ProLIF の既定。
   候補にする種類は `HBDonor, HBAcceptor, Anionic, Cationic, XBAcceptor, XBDonor, CationPi, PiCation,
   EdgeToFace, FaceToFace`。パッチは 2 つ: リガンドの化学構造を SDF から取り RDKit の芳香族性で読む、
   `Cationic` と `CationPi` の SMARTS で反対電荷の原子に結合した荷電原子を除く。
3. L1: 相互作用のリガンド原子 (`indices["ligand"]` の先頭) から、その原子、次に 3 結合以内の原子を
   探索順に見て、重原子・除外でない・結合相手 2 以上・重原子の結合相手 2 以上を満たす最初の 1 原子。
   除外は SMARTS `[#16](=O)=O` の 3 原子、`C(=O)[O!R]` と `C(=O)[O-]` の O 2 つ。
4. (L1, 残基, 種類) ごとにフレームを数え (1 フレーム 1 回)、`int(0.40 x 全フレーム数)` 以上を残す。
5. (L1, 残基) ごとに統合: フレームは和集合、種類は重み最大 (イオン 0.9、カチオン-π 0.7、水素結合 0.6、
   ハロゲン結合 0.4、スタッキング 0.1)。
6. タンパク質側は残基の主鎖 N, CA, C の 6 通りの並び (P1, P2, P3)。L2 は L1 に結合した重原子で末端で
   ない (結合相手が 1 つではない) もの、L3 は L2 に結合した同じ条件の原子 (L1 以外)。
7. 候補のフレームの中で、角度 L1-L2-L3 と P1-L1-L2 が 45 から 135 度 (両端を含む) のフレーム数が
   それぞれ `int(0.40 x 全フレーム数)` 以上の組だけ残す (L1-P1-P2 は条件にせず順位に使う)。
8. 順位: n = P1-L1-L2 と L1-P1-P2 それぞれの範囲内フレーム数。90 度からの平均ずれ (候補のフレームでの
   平均、度) の 2 角度の和が 40 以下で 1.0、60 で 0.7 まで直線、それ以上 0.7 を n に掛ける。二次構造が
   安定 (フレームの 50% 超) の残基に `3 x step` を足す。`step = max(10, 全フレーム数 // 50)`。
   点 = `50 - (最大の n - 値) // step` を 2 角度で合計し、種類の重みと、ボーナス 7 (その残基にイオン・
   カチオン-π・水素結合・ハロゲン結合の残った候補が 2 つ以上あり、フレーム数の 2 番目 / 1 番目 > 0.6) を
   足す。点の高い順、同点は原子番号順。二次構造は MDAnalysis の DSSP (H, E, - の 3 種)。
9. 距離による方法 (相互作用の候補が無い、または組が得られないとき): 3 の条件を満たすリガンド原子のうち
   N と O (無ければ P と S、次にハロゲン、次に Si と C を加える) と主鎖原子の、フレームごとの最近接対
   (最小像、10 A 以内) を残基ごとに集め、最頻のリガンド原子を L1 にする。`int(0.15 x 全フレーム数)` 以上の
   残基が候補。7 の条件を 15% で課し、(範囲内フレーム数の和 // `max(1, int(0.02 x 全フレーム数))`) の
   大きい順、90 度からのずれの和の小さい順。これも空なら計算を止める。
10. 参照値: 6 座標 (r = P1-L1、θ = L1-P1-P2、α = P1-L1-L2、二面角 3 つ) を全フレームで計算し、θ と α が
    ともに 45 から 135 度のフレームについて 20 ビンのヒストグラム (角度 0 から 180、二面角 -180 から 180、
    距離はデータの範囲) の最大ビンの中心。力の定数は 2 kcal/mol/A^2、80 kcal/mol/rad^2、80 kcal/mol
    (二面角は余弦型)。角度と二面角の 80 は cuFEP と同じで、距離だけ 2 と 10 で違う。

論文とソースの違い: 論文の「3 結合以内の原子から選ぶ」はソースでは条件を満たす最初の 1 原子。論文の
二次構造 5 種 (α、3-10、π ヘリックス、ストランド、ブリッジ) に対しソースの DSSP は 3 種。論文の
「カルボキシルの O」に対しソースの SMARTS は環外の O を持つカルボニル全般 (エステルも) に当たる。

ソースにある弱点 (cuFEP に持ち込まない): 残基を「残基名+番号」で区別する (鎖が複数あると衝突する)、
角度を周期像の補正なしで計算する。

訂正 (2026-11-21、手順 6 の実装中): 当初ここに「二面角の参照値を -180 から 180 のヒストグラムだけから
取るので ±180 度付近の山が両端のビンに割れる」を 3 つ目の弱点として挙げ、0 から 360 のヒストグラムと
比べて高い方を取る、としていた。これは誤り。2 つのヒストグラムはどちらも 18 度刻みで同じ角度に境界を
持ち、円周を同じビンに分けている。180 度の山が隣り合う 2 つのビンに分かれるのは、他のビン境界に
ある山と同じである。参照値は出典どおり -180 から 180 のヒストグラム 1 つから取る。

## Approach

設定: `restraints.selection: interactions | auto | manual`、既定 `interactions`。`auto` の意味は変えない。
新しい節 `restraints.interactions`: `trajectories` 4、`length_ns` 3.0、`discard_ns` 1.0、
`frame_interval_ps` 10 (論文 2.4 節); `min_frame_fraction` 0.40、`angle_range_deg` [45, 135]、
`stable_structure_fraction` 0.5、`second_interaction_ratio` 0.6、`fallback_frame_fraction` 0.15、
`fallback_cutoff_nm` 1.0 (論文 2.1 節)。ソースにだけある定数 (種類の重み、減衰 40/60/0.7、50 段階、
`3 x step`、ボーナス 7、2% 刻み、20 ビン、除外の SMARTS) は設定にせず名前つき定数とし、ソースの
ファイルと commit を `docs/configuration.md` に書く。

拘束用の MD は YAML の設定とバックエンドに従う。この MD だけの値は本数、長さ、捨てる長さ、保存間隔。
ほかは既存の設定を使う: `backend` (name、fallback、precision、platform)、`thermodynamics` (温度、圧力)、
`dynamics` (時間刻み、摩擦、barostat の頻度)、膜なら membrane barostat、`resources` (GPU、MPS、スレッド)、
`sampling.seed`。pre-equilibration の本番部分と同じ 1 つの関数を通し、実際に走ったバックエンドを記録する。

Felis と変えるところ (cuFEP の選択として `docs/configuration.md` に書く):
- MD は最小化直後からではなく pre-equilibration の終わりから。アルケミカル計算の開始点は今のまま。
- 4 本は交換なしの独立な軌跡。
- ProLIF はパッチなしの公開版を版固定で使い、SMARTS は引数で渡す。版は出力に記録する。
- 残基は残基 index で区別し、角度もリガンドを最近接像に移して計算する。
- 二次構造は論文の 5 種を mdtraj の DSSP (H, G, I, E, B) で判定する。
- 採った拘束は OpenFE の 2 規則も満たす (すべての拘束に課している基準を新しいモードにも適用)。
- 力の定数と関数形は cuFEP の設定のまま (距離 10、角度と二面角 80、調和型)。

手順 (小さな commit、各段で CPU の pytest が通る):

1. この計画書、PROGRESS、ドラフト pull request。Felis の `main` が d6dff91 のままか確かめ、Felis の
   テストが扱う場合分けを読む (読むだけ)。
2. 依存: `docker/environment.yml` に `prolif` (版固定)。イメージを作り直し、入れ替わったパッケージの版を
   記録し、コードを変える前に unit と GPU の全テスト。`docs/LICENSING.md`。
3. 平の MD の共通関数 (振る舞いを変えない): `_run_once` のバックエンドの選択と stock OpenMM での
   やり直し、`pre_equilibrate` の 2 fs でのやり直し、`run_leg` の設定を作る部分。
4. `backend.fallback: none` を平の MD でも守る (今は無視して stock OpenMM に切り替える)。回帰テスト。
5. `src/cufep/sampling/restraint_md.py`: `equil.npz` の最終状態から N 本、seed だけが違う。足りない軌跡が
   あるときだけ GPU を借り、借りた枚数だけプロセスを立てる (`hrex.py` の `start_workers`、`_Process`、
   `die_with_parent` と worker の環境の前処理を再利用)。終わったら借りた先頭の GPU で続ける。各プロセスが
   自分の軌跡を `restraint_md/trajectory_<k>.npz` に書く。記録 (`made_with`: MD の設定、seed、
   バックエンド、開始状態のダイジェスト) と検査。
6. アンカーから拘束を作る共通関数 (`manual` と共用)、参照値 (ヒストグラム)、残基ごとの DSSP の割合。
7. `src/cufep/alchemy/interaction_anchors.py`: 出典の 3 から 9。ホストは N、CA、C をちょうど 1 つずつ持つ
   アミノ酸残基。順位の高い候補から `anchor_geometry_problem` を通った最初の候補を採る (検査には
   解析フレームの後ろに開始配置を加える)。落とした候補は理由つきで記録。候補が尽きたら止める。
8. `src/cufep/analysis/interactions.py`: ProLIF の橋渡し (遅延 import、1 プロセス)。記録された時刻が
   `discard_ns` より後のフレーム、リガンドは最近接像、準備済みの分子を RDKit の芳香族性で読み直す、
   タンパク質は鎖全体、残基は残基 index。
9. 組み込み: 設定、`run_leg` の 3 分岐、何も書く前の拒否 (prolif が import できない、L1-L2-L3 の鎖を
   持たないリガンド、`discard_ns >= length_ns`、閾値のフレーム数が 0)、記録 (`restraint_made_with` は
   `interactions` 節をそのモードのときだけ記録し、以前の `restraint.json` を `auto` で再利用できるように
   する)、結果のキー (値はすべて `restraint.json` の `info` から)、`cufep plan` の表示、
   `docs/configuration.md`。この時点の既定は `auto`。
10. 出力: `interactions.json` と csv (相互作用ごとのフレーム数と頻度、全体と軌跡ごと; 残基ごとの二次構造の
    割合; 候補ごとの角度の頻度、点、順位)、図 `interactions.png`。
11. GPU テスト。
12. 既定を `interactions` に切り替える。`docs/protocol.md`、`docs/configuration.md`、
    `.claude/rules/fep-invariants.md`、README (英語と日本語)、`examples/README.md`、wiki (記録済みの
    ベンチマークは以前の既定で行ったと明記)、PROGRESS。
13. 書いていない者による diff 全体のレビューと修正。
14. commit からイメージを作り、`cufep check-env`、`tests/gpu`、TYK2 と A2A の example を各 1 回。結果を
    記録し、止めて報告する (ベンチマークの再実行はこの計画の外)。

範囲外: ベンチマーク full の再実行、`pre_equilibration.complex.md_ns` (5 ns) の短縮、距離の力の定数の
既定の変更、水を介した相互作用。

## Invariants

- 判定は公表された基準だけ。新しいモードの閾値 (40%、45 から 135 度、15%、1.0 nm、アンカーが無ければ
  停止) は Ding et al. 2026 の文を `docs/configuration.md` に引用する。ソースからの定数は順位づけ
  (計算の規則) で、判定ではない。相互作用の頻度などは数値として報告し、言葉の判定を付けない。
- すべての拘束が OpenFE の 2 規則を通るという既存の基準は、新しいモードでも変えない。
- cuPotential を「設定されているのに使われない」状態にしない。拘束用の MD も設定のバックエンドで走り、
  実際に走ったバックエンドを記録する。回復 (2 fs、stock OpenMM) はログと結果に書く。
- すべての設定に既定値と根拠。`auto` の意味は広げない。自動の延長・再実行はしない。1 リガンド 1 計算。
- cuPotential の速度と実行時間は追跡ファイル、commit、GitHub に書かない。
- 解析の数値は変えない (`ANALYSIS_VERSION` 10 のまま)。
- Felis は方法の出典として引用するだけ。そのコードはリポジトリに入れない。

## Verification

- CPU: `scripts/in_docker.sh python -m pytest -q`、`scripts/lint.sh`、harness の 3 スクリプト。
  - 規則: 合成した記録と座標で、各閾値の境界 (800 フレームで 320 は残り 319 は落ちる)、統合と重み、
    L1 の写像と除外、6 通りの並び、角度の条件の両端、点 (手計算の値)、同点、距離の方法、停止、
    OpenFE の規則で落ちる候補の飛ばし、挿入コードで区別される残基。
  - ProLIF: TYK2 と ejm_31 のヒンジの水素結合、n フレーム中 k フレームだけ結合がある軌跡で頻度 k/n、
    回転・並進・周期像で不変、五員の芳香環、残基ごとの形式電荷、ProLIF 自身の実行結果との一致。
  - 参照値: 分布を与えた合成データで最大ビンの中心、範囲外のフレームの除外。
  - 拘束用の MD: 積分した時間とフレームの時刻、各書き込みでの中断と再開、設定の違う軌跡の拒否、
    順番に走らせた場合とプロセスで走らせた場合の一致、2 fs と stock OpenMM でのやり直しの記録、
    `fallback: none` での停止、膜の圧力制御、設定が MD に渡ること。
  - ワークフロー: `auto` の記録が以前の版と同じ、`auto` のファイルを `interactions` で拒否、
    `cufep run`・`analyze`・`benchmark` で同じ値、実行前の拒否、設定と文書の一致。
- GPU: `scripts/in_docker.sh --gpus 0 python -m pytest tests/gpu -q`。拘束用の MD を両バックエンドで。
- 計算 (commit から作ったイメージ、各 1 回): `examples/kinase_tyk2/config.yaml` と
  `examples/membrane_a2a/config.yaml`。期待する証拠: PDB と SDF から手を入れずに結果まで終わり、同じ
  コマンドの再実行で何も計算し直さない; 拘束用の MD が設定のバックエンドで走ったと記録されている;
  `restraint.json` のアンカー・根拠の相互作用・参照値と、保存されたフレームから数え直した頻度の一致;
  文献の基準 (重なり 0.03、独立サンプル 50) の結果と ΔG ± 標準誤差を、実験値と以前の既定での結果に
  並べて数値で報告する (1 回の計算では精度の差は言えない)。

## レビュー (手順 13) の結果 (2026-11-21)

書いていない者が `main..f5a77ce` を読み、合法な入力で間違った数値が検査を通る例を探した。そのような例
(P1) は見つからなかった。Felis のソースから独立に書いた第 2 の実装との比較 (乱数で作った 1000 例) で、
選ばれるアンカーの違いは 0。指摘は P2 が 4 件、P3 が 4 件で、すべて再現してから直した
(commit 1739162、0db38f6、8b7ba95)。

- P2 周期像: 相互作用を数える前に、リガンドを「全 C-alpha の中心に最も近い像」に移していた。
  偏った複合体の外側の面にいるリガンドは中心から箱の半分より遠く、その像は反対側の溶媒の中になる
  (模型: 52 残基の領域と 477 残基の領域、箱 9.05 nm、中心から 4.93 nm。ProLIF に渡る座標で最近接の
  主鎖原子まで 1.21〜1.28 nm、MD が書いた座標では 0.30 nm 以下)。実在の複合体では再現していない
  (TYK2 は中心から 1.09 nm、箱の半分は 4.02 nm)。規則を 1 つにした (`ligand_next_to_protein`:
  リガンドの中心に最も近いタンパク質原子の像)。相互作用、`auto` の探索とその開始配置、入力ポーズからの
  RMSD、本番の構造の数値がこれを通る。中心の規則が正しかった場合は同じ像なので、数値は変わらない。
- P2 `--ligand` と `--leg`: 実行しないリガンド (または溶媒 leg だけを実行するリガンド) のせいで実行が
  拒否された。実行は、complex leg を実行するリガンドだけを調べる。`cufep plan` は全リガンドを調べる。
- P2 精度: `backend.precision: single` のとき、平の MD (pre-equilibration と拘束用の MD) は mixed で
  走っていた (最初の版からの cuFEP 独自の規則、出典なし)。`docs/protocol.md` は逆のことを書いていた。
  オーナーの決定 (2026-11-21): single と書けば平の MD も single で走る。以前の版が `single` の設定で
  作った `equil.npz` は mixed で走ったものなので、再利用しない (記録のキー `plain_md_precision`)。
- P2 `restraints.host_max_distance_nm`: 「`auto` だけ」と書いたが、どの選択でも
  `ligand_rmsd_from_input_A` を測る結合部位を決める。文書と schema の説明を直した。
- P3: `equil.npz` の記録に `backend.fallback` が無かった (`none` を記録する。以前の版のファイルは既定の
  ままなら使える)。スケジューラに伝える仕事量の単位が部分ごとに違った (`md_work` に統一) とログの文言。
  軌跡のプロセスが結果を送って終わる瞬間の読み落としと、終わらないプロセスの放置。2 本の鎖で同じ番号の
  残基が同じ名前になること (鎖が複数あれば `VAL981.B`)。`restraint.json` の開始配置の値の単位。

## Codex によるレビューの結果 (2026-11-21)

オーナーの指示で、push した head (`eb6f89f`) を Codex (gpt-6.2-astra、effort ultra) にもレビューさせ、
PR #6 に原文のまま投稿した (行コメント 26 件: P1 11、P2 13、P3 2。Docker に届かないため、すべて読んだ
だけの指摘)。各指摘をコードで再現してから直した (commit b4b26e6、9b0f8fd)。

- 周期像と規則: 斜めの箱での最近接像 (`nearest_image`)、拘束の 6 座標を力が測る像 (G0 が H0 に最も
  近い像) で測ること、開始配置は 1 配置の規則だけに掛け「全フレームで一直線」は解析フレームに掛ける
  こと、`force_constants: auto` では揺らぎから決まる力の定数で平坦な角度の規則 (OpenFE) を確かめ直す
  こと (与えたアンカー、`auto` の探索、`interactions` の 3 つとも)、参照値を作れない候補は飛ばすこと。
- 閾値と記録: フレーム数の閾値は設定値を書かれたとおりの小数として切り捨てる (800 の 0.29 は 232)。
  記録の整数は厳密に比べる。`interactions` の拘束は軌跡の開始状態に結び付ける。pre-equilibration の
  記録に、実際に走った精度 (プラグインに double を指定した場合) と platform。構造の数値に版。
- プロセスとバックエンド: 公開済みの軌跡はプロセスが落ちても使う。バックエンドには開始時の箱の System を
  渡す。平の MD に `resources.cpu_threads_per_worker`。
- 報告: 結果は名前を出した相互作用のフレーム数と、その原子・残基の採択済み全種類のフレーム数を分けて
  記録する。残基名は入力 PDB の呼び方 (鎖は名前が重なるときだけ)。図は leg の拘束のものだけ。
  `cufep analyze` に `restraint_choice`。棄却した候補はすべて `interactions.json` に残す。
- 変えなかった指摘 (理由は PR に返信): タンパク質の鎖が別の周期像にある場合 (cuFEP の MD は座標を
  折り返さない。A2A の 1200 フレームで確認)、計画時の stock OpenMM への切り替えの記録 (この PR より前
  からの挙動で、実行の記録に残る)。閾値が 0 フレームに丸まる設定の拒否は残す (オーナーの決定、
  2026-11-21: 「拒否のままでいい」。論文が定めるのは切り捨てだけで、拒否は cuFEP の決まり)。

## Codex による再レビューの結果 (2026-11-21)

修正後の head (`877ab88`) を Codex に再レビューさせた (PR のレビュー 6003995828、行コメント 36 件:
P1 22、P2 11、P3 3。すべて読んだだけの指摘)。再現してから直したものは commit 88965b8。

- 拘束と力: 拘束の 6 座標は、力 (OpenMM) と同じ周期処理で測る (`as_the_force_measures`。6 原子を
  1 つ前の原子からの周期的な変位で並べる)。平坦な角度の規則 (OpenFE) は 1 回だけ、拘束が実際に持つ
  力の定数で、全フレーム・開始配置・与えたアンカーの準備構造・参照角そのものに掛ける。
- 相互作用からの選択: リガンドの像はタンパク質の全原子のうち最も近い原子で決める。距離による方法は
  出典どおりキャップの N/CA/C も見る。0 フレームの閾値の拒否を実測のフレーム数にも掛ける。
- 記録: 設定の比較を厳密にする (許容差なし)。pre-equilibration の platform は常に記録し、記録の無い
  旧ファイルは従来どおり使う。拘束と開始状態の対応を、標本を読むすべての経路・ジョブ開始前・図で
  確かめる。揺らぎからの力の定数を持つ旧版の拘束は再利用しない。
- 変えなかった指摘 (理由は PR に返信): 3 (`auto` の探索の周期像。この PR より前からのコードで、既定の
  距離範囲では起きない)、8 (厳密に一直線のフレーム)、14 (旧版の `ligand_rmsd_A`)、22 (1e-9 ps の
  許容差)、24 (1 本だけの軌跡のプロセス分離)、26 (失敗した途中の試行の注記)、35 (201 件のテスト)。

オーナーの指示 (2026-11-21): 修正が必要なものだけ直す。再レビューの第一の目的は修正が完了したかの
確認で、その旨を Codex への指示に入れる。

## example の実行結果 (手順 14、開始点の変更の前)

どれも既定の `restraints.selection: interactions`、本番 5 ns、GPU 4 枚、1 回の計算。拘束用の MD は
pre-equilibration (拘束なしの MD 5 ns、膜では 10 ns) の終わりから始めた版である。

| TYK2 ejm_31 (実験 -9.54) | イメージ | ΔG_bind | complex leg | ホスト | 水素結合 |
|---|---|---|---|---|---|
| 1 回目 | `0f3aa32` | -10.85 ± 0.65 | 41.47 | VAL981 N, C, CA | 672/800 |
| 同じイメージでもう一度 | `0f3aa32` | -11.99 ± 0.52 | 42.42 | VAL981 N, CA, C | 723/800 |
| 2 回目 | `fce6929` | -5.83 ± 0.75 | 36.33 | VAL981 N, C, CA | 717/800 |
| 3 回目 | `88965b8` | -9.77 ± 0.60 | 40.52 | VAL981 N, CA, C | 751/800 |

| A2A 4e (実験 -12.07) | イメージ | ΔG_bind | complex leg | ホスト | 水素結合 |
|---|---|---|---|---|---|
| 1 回目 | `0f3aa32` | -12.14 ± 0.49 | 38.71 | ALA59 | 773/800 |
| 2 回目 | `fce6929` | -16.85 ± 0.64 | 43.04 | ASN253 | 787/800 |

- TYK2 の 4 回: 平均 -9.61、標準偏差 2.7 kcal/mol (-5.83 を除くと -10.87、1.1)。同じイメージの 2 回の差は
  1.1。`fce6929` の変更をすべて含む `88965b8` は -9.77 なので、-5.83 は修正による変化ではなく実行間の
  ばらつきと見る。ゲスト原子は 4 回とも同じ。重なりの基準 (0.03) は全回で満たし、NOT RELIABLE は無い。
  1 回の計算の標準誤差 (0.5〜0.75) はこのばらつきを表していない (issue #5)。
- 3 つの版のコード (`0f3aa32`、`fce6929`、`88965b8`) で、保存した軌跡から拘束を選び直すと、TYK2 の 2 回、
  A2A の 2 回とも保存された拘束と数値まで一致する。2 つの commit の間でレプリカ交換・アルケミカル系・
  解析のコードは変わっていない。
- A2A の 2 回は、拘束用の MD の時点でリガンドの相互作用が違う。1 回目は ALA59 (773)、HIS278 (772)、
  GLU169 (755) で、入力ポーズにある ASN253 との水素結合が残っていない。2 回目は ALA59 (800)、ASN253
  (794 と 787)、HIS278 (714)。10 ns の拘束なしの MD の間に、リガンドが 2 回で別の状態に落ち着いた。

## 拘束用の MD の開始点を出典に合わせる (オーナーの決定、2026-11-21)

根拠: 上の A2A の 2 回。拘束を測る前の拘束なしの MD (5 ns、膜では 10 ns) の間にリガンドがどの状態へ
動くかが実行ごとに違い、その状態が拘束として固定されていた。出典 (Ding et al.) は最小化した入力構造から
4 本を始め、アルケミカル計算も最小化した入力構造から始める。

`restraints.selection: interactions` の complex leg だけを変える (`pre_equilibration_parts`):

1. 水溶性の系: 最小化のあと、タンパク質とリガンドの重原子を位置拘束したまま NVT 0.25 ns と NPT 0.5 ns
   を走らせ、そこで pre-equilibration を終える。拘束なしの 5 ns は走らせない (4 本の軌跡がその役割)。
2. アルケミカル計算の開始点は、4 本と同じ pre-equilibration の終わり (入力ポーズのまま、水と箱が緩和した
   状態)。拘束が開始配置で OpenFE の規則を満たすことは今までどおり確かめる。
3. 膜の系: 作ったばかりの二重膜を落ち着かせる `membrane.equilibration_ns` (10 ns) は必要なので残し、
   その間も溶質の重原子を位置拘束する (水を膜の中心から遠ざける壁は NVT と NPT の間だけ)。

- `selection: auto` と `manual`、溶媒 leg は変えない (拘束なしの MD のフレームから拘束を決めるため)。
- `equil.npz` の記録に `solute_held` を入れる。以前の版のファイル (拘束なしの MD) は `interactions` では
  使わない。
- `ligand_rmsd_from_input_A` は、この選択では拘束用の MD の解析フレーム (等間隔に最大 200) で測る。
- 距離の力の定数は既定の 10 kcal/mol/Å² のままにする (オーナーの決定、2026-11-21)。4 回の実行の拘束なしのフレームから
  求めると、結合状態で拘束をかけるコストは 6 項で 1.5〜2.5 kcal/mol、そのうち距離の項は 0.12〜0.21
  (定数 2 なら 0.03〜0.06) で、大半は角度と二面角 (出典と同じ 80) による。距離の揺らぎは 0.18〜0.23 Å、
  拘束の幅は定数 10 で 0.24 Å、2 で 0.54 Å。
- TYK2 の実行間のばらつき (拘束が同じでも complex leg が動く) は、この変更では説明も解決もできない。


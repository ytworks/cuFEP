# RBFE の精度改善(issue #27): 窓の開始、末端回転 MC、ダミー原子の結合項、REST2 の hot 原子
Status: In progress
Refs: issue #27; this pull request; `docs/rbfe.md`; `docs/configuration.md`; `benchmarks/results/{tyk2,a2a}_rbfe/`

## Context
- オーナーの依頼(2026-11-28): issue #27 の仮説を読み、精度改善の実行計画を立てて進める。目的は issue の通り、入力構造・
  原子マップ・ネットワーク・実験値を固定し、1 変換の所要時間を増やさずに、生の辺 ΔΔG の誤差を下げること。
- いまの記録(イメージ 957a9cf、1 辺 1 計算、kcal/mol): TYK2 22 辺の RMSE 0.89(同じ 22 辺で OpenFE 0.61)、A2A は公開 map と
  共通の 10 辺で 2.16(FEP+ 1.42)。
- オーナーの決定(2026-11-28): 検証 run は各条件 2 回。GPU は 0〜5 の 6 枚。留保する標的(issue の「標的ファミリー全体の
  留保」)は、既定にしたい変更が出てから決める。
- 用語: **辺** = 1 つの変換 A→B。**対照** = 変更を off にした計算、**変更側** = on にした計算。**ダミー原子** = 片方の
  リガンドにしかなく、反対の端では非結合相互作用が消えている原子。**接合部** = ダミー原子が、両方のリガンドにある原子
  (core)につながる所。**hot 原子** = REST2 で相互作用を弱める(加熱する)原子。

## Approach
issue の項目はどれも設定として実装し、既定はオーナーが決めるまで変えない(既定 = いまの動作)。

| 順 | 項目 | 設定 | 実装の要点 | 最初の検証 run |
|---|---|---|---|---|
| 1 | 窓数の対照(11 窓固定)と、対照に必要な道具 | `lambda_schedule.start_configurations: pre_equilibration \| precalculation`、窓の値と `sampling.production_ns` を脚ごとに | 与えた窓にも、適応 λ の 1 回目と同じ作り方(walk のあと、各窓で precalculation と同じ長さの MD)で開始構造を作る。walk は 1 つの関数にして両方が使う。脚の MD の内訳を結果に書く | TYK2 `ejm_55~ejm_54`(issue が挙げた辺): 記録の窓(9/9) 対 等間隔 11 窓。開始の作り方と脚ごとの総 MD は同じ。4 run |
| 2 | 末端回転 MC(issue の優先 1) | `sampling.terminal_flip.enabled: false`、`interval_ps: 2.5` | 新 `src/cufep/sampling/terminal_flip.py`。水の MC と同じ場所(worker の MD の直後)で、その λ/REST2 状態の全ポテンシャルで Metropolis 判定。環に属さない結合を軸に、その先の芳香環の末端基(両リガンドの固有原子を含む)を 180° 剛体回転し、速度も同じ回転。向きが区別できない対称な環、片端だけで閉じる環、軸外への拘束がある基は対象外 | 対象の辺 1 つ × {off, on} × {入力の向き, 反転した向きから開始}。4 run |
| 3 | ダミー原子の結合項(優先 2) | `alchemy.dummy_bonded_terms: all \| separated` | 新 `src/cufep/rbfe/junctions.py` が結合グラフから接合部を分類する(Fleck, Wieder, Boresch, J. Chem. Theory Comput. 2021, 17, 4403 の terminal / dual / 平面 triple)。余分な角度・二面角は、既存の補間する力に「ダミー側の端で力の定数 0」として入れる。分離を示せない形(両側にまたがる環、非平面 triple)は今のまま残し、記録に出す | 小分子の相対溶媒和サイクル → 対象の辺 1 つ × {all, separated} × 2 回。4 run |
| 4 | REST2 の hot 原子(優先 3) | `sampling.rest2.region` に `perturbed` を追加(RBFE。既定は `pocket` のまま) | hot = 固有原子と、両端で力場の項が変わる core 原子だけ。受容体は加えない。リガンド内の LJ・静電に対ごとの hot 数、エネルギー評価(`src/cufep/rbfe/states.py`)に群を 1 つ追加 | 対象の辺 1 つ × {pocket, perturbed} × 2 回。4 run。REST2 なし(scale 1)は条件つき |

- 項目 4: 電荷は化合物ごとに付けるので、core の電荷はほぼ全部がわずかに変わり、「項が変わる原子」をそのまま取るとリガンド
  全体になりうる。辺ごとの hot 原子数を先に出し、「変わる」の定義は出典(Liao et al., J. Chem. Theory Comput. 2024,
  doi:10.1021/acs.jctc.4c00954)を読んで決める。
- 無条件の run は 16(1 辺ずつ)。残りの対象辺、TYK2・A2A の全体、留保標的は、各報告のあとオーナーの指示で行う。
- 出典は実装の前に全文を読み、規則を `docs/rbfe.md` と `docs/configuration.md` に引用する: 末端回転 MC
  (doi:10.1021/acs.jctc.6c00177、doi:10.1021/acs.jctc.6c00593、SOMD2 の `_terminal_flip.py`)、Fleck et al. 2021、Liao et al.
  2024、Baumann et al.(doi:10.1021/acs.jcim.6c00089)§3.3。文献にない選択は cuFEP 独自と書く。

### 手順
1. 項目 1 を実装する。記録の 43 辺の分類表を作る(実験値を使わない。GPU 不要): 末端回転の対象基、接合部の型、hot 原子数
   (pocket と perturbed)。保存フレーム(λ = 0 の溶質)から対象基の二面角の集団も出す。表と「最初の辺」の案を報告し、
   オーナーの確認を得てから run を始める。
2. 項目 1 → 2 → 3 → 4 の順に、実装 → CPU テスト → push → 検証 run → 1 辺ごとに報告。run の間に次の項目を実装する。
3. 全項目の結果を 1 つの表にして報告する。既定にするか、留保標的を追加するかはオーナーが決める。

### 検証 run の条件(全項目で同じ)
- イメージ: この pull request のコミットから作った `cufep:<commit>` を、対照と変更側の両方に使う。run の前に、設定 off の
  hybrid System の checksum と samples の設定が、957a9cf の記録と同じことをコードと記録で示す。
- 入力: 記録の `resolved_config.yaml` と同じ蛋白質・リガンド・マップ・辺。窓は、記録でその辺の各脚が選んだ値を両側に
  与え、開始構造は項目 1 の `precalculation`。REST2・GCMC・交換間隔・production は記録と同じ。
- 時間: `timing.json` の壁時計と GPU 時間を両側で測る。変更側が対照を超えたら、その分 production を短くして測り直す
  (自動にはしない)。
- 報告する数字: 辺の ΔΔG と実験との差、2 回の差、対象の二面角や core の内部座標の集団、状態間の往復、overlap と独立
  サンプル数、MC の試行と受理、適用された辺の数。適用外の辺は記録の値のまま集計に入れる。
- issue の通り、次の変更は既定にしない: 回転や交換が増えても集団と辺の誤差が変わらないもの、時間が対照を超えるもの、
  失敗した辺を除いて初めて良くなるもの。

## Invariants
- 判定は文献の基準だけ(overlap 0.03、独立サンプル 50)。この計画の比較は数字で報告し、判定の言葉を付けない。
- 新しい設定はどれも既定でいまの動作になり、既定でないときだけ samples の設定に記録する。いまの脚の記録と解析の数字は
  変わらない(`ANALYSIS_VERSION` は変えない)。
- 1 辺 1 計算の既定は変えない。2 回の計算は検証のためだけに行う。
- プラグインの速度と時間は、追跡するファイルにも GitHub にも書かない。

## Verification
- 項目 1: 与えた窓の開始が適応 λ の 1 回目と同じ作り方になること、脚の MD が precalculation を含めて数えられること。
- 項目 2(`tests/unit/test_terminal_flip.py`): 回転と逆回転で元の配置に戻り比が逆になる、MC だけで深さの違う 2 つの井戸の
  二面角の集団が exp(-βΔ) になる(Reference プラットフォーム)、拘束と運動エネルギーが保たれる、片端だけで閉じる環は
  対象外、両リガンドのダミーが環と一緒に回る。
- 項目 3: ダミーの座標を求積で積分した値が core の配置によらない(`all` では変わる = 修正前に失敗する)、端点の core の
  角度の分布がリガンド単独と同じ、キラル中心のダミーが反対側に行かない、両脚の接合部が同じ。
- 項目 4: hot–hot は s、hot–cold は √s、cold–cold は 1 になることを別に作った System と照合、両端は元のポテンシャル、
  エネルギー評価が各状態を直接設定した値と一致、両バックエンドの一致(GPU)。
- 実行: 上の表の検証 run(コミットから作ったイメージ、GPU 0〜5、1 辺ずつ)。

## Record

### 2026-11-28: 項目 1 の実装(f80b132)
- `lambda_schedule.start_configurations: pre_equilibration | precalculation`(既定は前者 = いまの動作)、`lambda_schedule.windows`
  と `sampling.production_ns` を脚ごとに、脚の結果に `md_ps`(pre-equilibration のあとの MD: 窓を作った MD と replica
  exchange)。walk は `cufep.alchemy.schedule.walk_starts` の 1 つを、適応 λ の 1 回目と、与えた窓の開始
  (`precalculation_starts`)の両方が使う。
- テスト: 与えた窓の開始構造が、同じ窓を選んだ precalculation の開始構造とビット単位で一致する(CPU、1 スレッド、決定的な
  力。`test_given_windows_started_as_a_precalculation_start_as_its_first_iteration_does`)。対象のテスト 80 passed、GPU テスト
  70 passed(GPU 2 枚)。
- 記録と同じ System になること(`spikes/m39_same_system_as_a_record.py`、run の前の確認 `check_leg_reuse` を記録の
  ディレクトリに対して行う): `runs/rbfe_bench_tyk2_957a9cf` の 44 脚と `runs/rbfe_bench_a2a_957a9cf` の 42 脚のすべてで、
  このツリーが作る hybrid System の checksum・窓・samples の条件が記録と同じ(違いは 0 脚)。

### 2026-11-28: 記録の 43 辺の分類(実験値を使わない。`spikes/m40_rbfe_edge_classes.py`、`spikes/m41_terminal_group_orientations.py`)
- 末端回転 MC の対象基がある辺: TYK2 は 22 辺中 0(2,6-ジクロロフェニルは 2 つの向きが同じ。ピリジン側は結合の大きい
  側)。A2A は 21 辺中 13(4b~4e、4b~4m、4f~4e、4g~4j、4g~4k、4g~4l、4h~4l、4h~4r、4j~4i、4k~4r、4m~4n、4m~4q、4o~4e。
  回る原子は 10〜18)。公開 map が回転異性体の状態を置いた化合物(4b、4e、4j、4k、4l、4m、4r)を含む辺と一致する。
- 保存フレーム(λ = 0 のレプリカ、各脚 600 フレーム)での対象基の向き: 13 辺すべてで、複合体の脚は開始と反対の向きが
  0.0 %(フレーム間の入れ替わり 0 回)。溶媒の脚は 6.7〜44.2 %(入れ替わり 68〜254 回)。
- 接合部: 橋の原子の混成が両端で変わるダミー基がある辺は TYK2 の 3 辺(ejm_42~ejm_54、ejm_50~ejm_55、jmc_28~jmc_30)、
  A2A は 0。
- REST2: いまの hot 原子はリガンド全体(31〜47 原子)と受容体 190〜278 原子。固有原子は辺あたり 0〜18、元素が変わるか
  固有原子に結合する core 原子は 1〜4、電荷が 0.05 e より大きく変わる core 原子は 0〜7。

### 2026-11-28: 項目 2 の実装(3367749)
- `sampling.terminal_flip.{enabled: false, interval_ps: 2.5}`、`src/cufep/sampling/terminal_flip.py`。規則と SOMD2 との
  違いは `docs/rbfe.md`("Sampling")。対象基がない脚は設定なしと同じに計算し、samples の設定にも書かない。
- テスト(`tests/unit/test_terminal_flip.py` 15 件、`tests/unit/test_rbfe_run.py` 2 件): 規則(小さい側の芳香環、向きの
  区別、両リガンドの基、ダミーも回る、対象外の理由)、半回転 2 回で元に戻る、MC だけで 1 kT 差の 2 つの向きの集団が
  0.269 になる(Reference、6000 回)、受理で速度も回り拘束長と運動エネルギーが保たれ、棄却で何も変わらない、脚で 1 反復
  1 試行が窓ごとに記録され、設定なしの run はその samples を引き継がない。
- GPU テスト(`tests/gpu/test_gpu.py`、8 件): 両バックエンドの Context で、leg の中間の状態(REST2 あり・なし)の MD の
  あとに半回転を 1 回試み、判定に使ったエネルギー差が、標準 OpenMM・倍精度・Ewald 誤差 1e-6 の同じ状態での 2 つの配置の
  差と一致する(12 回の測定で差は最大 0.006 kT、エネルギー差は 0.0〜10.0 kT、受理と棄却の両方。水準 0.05 kT)。

### 検証 run: 未実行
- 項目 1 の対照(TYK2 `ejm_55~ejm_54`)の入力は用意した: `runs/rbfe_acc_item1_recorded_windows_f80b132/` と
  `runs/rbfe_acc_item1_11_windows_f80b132/`(記録の `prep/` と `cache/` を写したもの、イメージ `cufep:f80b132`)。
  両側とも `mode: fixed`、`start_configurations: precalculation`、`sampling.repeats: 2`。記録の窓の側は両脚
  [0, 0.12, 0.25, 0.38, 0.5, 0.62, 0.75, 0.88, 1.0]、production 5 ns。11 窓の側は等間隔 11 窓で、production は
  複合体 3.725 ns、溶媒 3.8975 ns(脚ごとの MD の合計が記録の窓の側以下になる長さ)。オーナーの確認のあとに始める。

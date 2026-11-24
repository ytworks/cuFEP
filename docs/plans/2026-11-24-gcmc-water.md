# 水の grand canonical Monte Carlo (GCMC) を複合体レグに入れる
Status: In progress (2026-11-25: 承認。リスク確認(水の相互作用を実行中に切り替える)は通った。出典の全文の確認と実装はこれから。既定は off)
Refs: this pull request (#15); `docs/protocol.md` ("Water relocation moves", "Limits of version 0.1"); `docs/configuration.md` (`sampling`); `docs/plans/2026-11-24-opc-water-model.md`; `docs/plans/2026-11-22-accuracy-causes-and-fixes.md` (E4)

## Context

- A2A は結合が強く出すぎる(e77a6c5 の run の平均符号付き誤差 -4.14 kcal/mol)。リガンドを消した側の窓で、
  リガンドのあった場所の水の数が 20〜30 ns かけて増える(計画 2026-11-22)。
- 粒子数を保つ水の relocation の移動(RelocationMC、`sampling.relocation_mc`)は、4k で結果を変えなかった
  (二重層 + OPC: 移動あり -19.16 ± 0.59、なし -19.15 ± 0.35。受理は球の中へ 0.6 %、外へ 0.3 %。
  計画 2026-11-24 の OPC の記録)。
- オーナーの指示(2026-11-24): その run を止め、GCMC を適切に適用する。調べた結果、cuPotential プラグイン 0.3.5 と
  OpenMM 8.4 に GCMC は無く、OpenMM 用の GCMC(grand、loch、GrandFEP)は Python のパッケージである。オーナーの回答:
  cuFEP に実装する。水は TIP3P と OPC の両方に対応する。
- 計画 2026-11-22 は GCMC を「粒子数が変わり、今の座標の持ち方、交換、MBAR に合わない」として見送った。下の方式
  (ghost の水)は系の粒子数を変えないので、その理由は当たらない。

## Approach

出典は実装の前に全文を読み、規則を `docs/configuration.md` に引用する。出典が書いていない点は cuFEP 独自として記録する。

- **ghost の水**(grand: Samways, Bruce Macdonald, Essex, J. Chem. Inf. Model. 2020, 60, 4436): 系の粒子数は変えない。
  相互作用を切った水(ghost)を持っておき、挿入は ghost を球の中に置いて相互作用を入れ、削除は球の中の水の相互作用を
  切る。ghost は、準備した系の水のうちリガンドから遠いものを使う(独自。準備した系は変わらない)。ghost が尽きたら
  grand と同じく理由を出して止める。
- **球と受理**(grand のソース `samplers.py` で確認): 球はリガンドの重原子の中心。
  挿入 min(1, e^B e^(-βΔU) / (N + 1))、削除 min(1, N e^(-B) e^(-βΔU))、B = βμ'ex + ln(V_球 / V°)。
  N は酸素が球の中にある水の数。挿入の位置は球の中に一様、向きは一様。
- **体積一定**: grand は barostat があると止まる("GCMC must be used at constant volume")。GCMC を入れた複合体レグは、
  今の NPT の平衡化(二重層は 10 ns)と拘束のための MD のあと、箱を平衡化の後半の平均に固定して(独自)、窓の決定
  (precalculation)と本計算を行う。溶媒レグは今のまま(NPT、GCMC なし)。
- **試行をまとめて評価**(Ross et al., J. Chem. Theory Comput. 2020, 16, 6061。loch が同じ方式を実装): 1 回ずつの
  挿入・削除の受理は 0.01〜0.03 %(Melling et al., J. Chem. Theory Comput. 2023, 19, 1050)。今ある GPU カーネル
  (Lennard-Jones と reaction field、`src/cufep/sampling/relocation_mc.py`)で多数の試行を一度に評価し、通った最初の
  試行を、シミュレーションのエネルギー(プラグインの PME)で判定し直す
  (Gelb, J. Chem. Phys. 2003, 118, 7747: min(1, exp(-β[(U(j) - U(i)) - (U'(j) - U'(i))])))。
- **HREX と MBAR**: 各レプリカが水の on/off の一覧を持って窓を移る。-μN の項はどの窓でも同じなので、交換の判定と
  MBAR の式は変わらない。REST2、adaptive lambda scheduling、解析の判定(overlap 0.03、独立なサンプル 50)はそのまま。
- **μ'ex と V°**: TIP3P の公表値は -6.09 kcal/mol と 30.345 Å³(grand の既定値)。cuFEP の設定(カットオフ 0.9 nm、
  PME、分散補正)での値を、水 1 個の消去(cuFEP の HREX と MBAR)と NPT の密度で求める。公表値と誤差の中で合えば
  公表値を使い、合わなければオーナーに報告する。OPC は公表値が見つかっていないので、同じ手順の値を使う(独自)。
- **設定**: `sampling.gcmc`(既定 off)。`sampling.relocation_mc` との併用、`analysis.lrc: isotropic`、μ'ex と V° の無い
  水モデルは、何も書く前に設定の段階で拒否する。
- 最初の版でやらないこと: 非平衡の挿入(GCNCMC)、箱全体の GCMC、溶媒レグの GCMC、既定値の変更。

実装する場所:

- `src/cufep/sampling/gcmc.py`(新規)。球、stand-in のカーネル、フックは RelocationMC と共有の部品にする。
- `src/cufep/backends/base.py`、`cupotential.py`、`openmm_stock.py`: 水のパラメータを Context の中で切り替える入口。
- on/off の一覧: `sampling/worker.py`(共有メモリ)、`sampling/hrex.py` と `sampling/storage.py`(checkpoint)、
  `alchemy/schedule.py`(precalculation)、`workflow.py`(窓の開始構造、平衡化の出力、記録)。
- 体積一定の複合体レグ(`workflow.py`)。`alchemy/lrc.py`(tail correction)と `validate.py` は ghost を除く。

## Invariants

- 新しい判定は足さない。報告は数値だけ: 窓ごとの挿入・削除の試行数と受理数、球の中の水の数。
- 既定値は変えない(`sampling.gcmc.enabled: false`、水は TIP3P)。設定は全て YAML で指定でき、既定値と根拠を
  `docs/configuration.md` に書く。
- 入れた機能が「on なのに働かない」設定にしない: 受理が 0 のままになる試行数を既定にしない。
- 解析の数値は変えない(`ANALYSIS_VERSION` 10)。GCMC の無いレグの結果は 1 ビットも変わらない。
- 公開されているコードのうち GPL のもの(loch)は文書だけを読み、コードは写さない。
- cuPotential プラグインの速度や実行時間は、追跡するファイルに書かない。

## Verification

- unit(CPU、stock OpenMM): ghost にした水のある系のエネルギーと力が、その水の無い系と一致する(全ての窓)。
  相互作用の無い水で、球の中の水の数が平均 e^B の Poisson 分布になる。挿入と削除の比が互いに逆数。on/off の一覧つきで、
  書き込みの各段階での中断と再開。`cufep run`、`cufep analyze`、`cufep benchmark` で同じ結果。拒否は何も書く前。
- GPU: ghost のエネルギーがプラグインと stock OpenMM で基準(倍精度、Ewald 1e-6)に合う。GCMC つきの短い HREX が走り、
  再開できる。バルクの水を μVT で流し、球の中の水の数の平均が V_球 / V° になる(TIP3P と OPC)。
- run: コミットから作ったイメージで、A2A を 1 本ずつ GPU 8 枚。4k(TIP3P)、4k(OPC)、4e(TIP3P)、4e(OPC)の順に
  1 本ごとに報告する。残りをどちらの水で流すかはオーナーが決める。比較対象は実験値と、e77a6c5(二重層 + TIP3P)、
  0e3b2a9(二重層 + OPC)、9989cdb(二重層 + OPC + RelocationMC)の run。

## 記録

### 2026-11-25: リスク確認。水の相互作用は実行中に切り替えられる(`spikes/m33_ghost_waters_on_the_backends.py`)

系は GPU テストの REST2 の系(アラニンジペプチド 2 個、3.2 nm の水の箱、1 個がリガンド、ソフトコアと REST2、分散補正あり)。
水の各サイトについて、NonbondedForce(cuPotential ではプラグインの力)の電荷と epsilon、ソフトコアの力の epsilon を 0 にし、
`updateParametersInContext` で Context に書いた。TIP3P(水 1,026 個)と OPC(1,034 個、4 サイト)。

| 確認 | stock OpenMM | cuPotential |
|---|---|---|
| ghost は「無い水」か(stock、倍精度: 作るときに切った系と、その水の無いトポロジーから作った系) | エネルギーの差 0.0004 kJ/mol(43,334 のうち)、力は相対 1e-12 | – |
| 実行中に切った Context と、切って作った Context(3 つの窓、力の群ごとのエネルギー、力、全ての窓の reduced potential) | 0.005〜0.009 kJ/mol、力は相対 1.4e-6〜1.6e-6(単精度の範囲) | 差 0(全ての数値が一致) |
| 入れ直した Context と、切り替えていない Context | 同上 | 差 0 |
| 窓の切り替えと混ぜた 40 回の切り替えのあと | 同上 | 差 0 |
| ghost を別の位置に置いて入れた Context | 同上 | 差 0 |
| ghost のある MD 250 ステップのあと(エネルギーは有限) | 同上 | 差 0 |
| 6 個の水を切ったときのエネルギーの変化、TIP3P(基準: 倍精度、Ewald 1e-6 で 406.795 kJ/mol) | 406.783 | 406.787 |
| 同、OPC(基準 518.119 kJ/mol) | 518.698 | 518.148 |

- 結論: 水の on/off は `setParticleParameters` と `updateParametersInContext` で行う。両方のバックエンドで、切り替えた
  Context は「そう作った系」と同じ数値を出し、分散補正も切り替えに追従する(分散補正なしの系では変化が 404.085 に
  なり、差 2.710 kJ/mol が基準とプラグインで一致)。global parameter と offset を使う代わりの方法は要らない。
- OPC での stock OpenMM の 0.58 kJ/mol のずれは、既に記録した stock の PME の誤差(Ewald 5e-4)である。

**この確認で分かった別のこと(GCMC とは別)**: REST2 と分散補正の両方がある系では、窓のあいだのエネルギーの差が、
プラグインと基準で最大 0.55 kT(TIP3P)、0.68 kT(OPC)違う。窓ごとの差は REST2 の強さに沿う
(TIP3P、プラグイン - 基準: 0、0.295、0.452、0.553、0.480、0.382、0.237、-0.006 kT)。REST2 なしでは 0.006 kT(TIP3P)と
0.009 kT(OPC)、分散補正なしでも同じ値。REST2 が epsilon を下げる原子の分散補正の扱いが、2 つのバックエンドで違う。
符号(REST2 の強い窓でプラグインのエネルギーが高い)は、プラグインが分散補正も下げ、stock OpenMM は下げない場合に合う
(どちらがそうしているかは、まだ直接には確かめていない)。
run は分散補正あり(`src/cufep/prep/forcefield.py`)、GPU テストの系は分散補正なしなので、テストはこれを見ていない。
この項は配置によらず(体積だけに依存)、REST2 の強さが 1 の両端では 0 なので、交換の判定と両端のあいだの自由エネルギーは
変えない。対処(テストを分散補正ありにする、プラグインと stock の差として記録する)は、オーナーに報告してから決める。

### 2026-11-25: 出典の確認

書いた本人でない読み手が出典を集めて規則を書き出した(引用つき)。読めたもの: Ross 2017(本文の全部、式 1〜4、6、7)、
grand のソース(`samplers.py` の基底クラス、球のサンプラー、標準のサンプラー、`utils.add_ghosts`、例)、loch の white paper
(全部)と README、Melling 2023(理論と方法の部分)。取得できなかったもの: Ross 2015(JACS)、grand の論文の本文
(Samways 2020)、Ross 2020(要旨だけ)、Gelb 2003。試行をまとめて評価する方式と 2 回目の判定の式は、loch の white paper の
記述を出典にした(`docs/configuration.md` にその旨を書いた)。

実装に効いた点:

- 受理の式と B は Ross 2017 の式 1、2、7(grand のソースと同じ)。Ross 2017 は 2015 の平衡の条件を「erroneously stated」と
  訂正しているので、2017 の式 7 を使う。
- N は「ghost でない水で、酸素が球の中にあるもの」。呼ぶたびに位置から数え直す(MD で出入りした水も数える)。球の中心は
  基準の原子に付いて動く。N = 0 の削除は何もしないが試行として数える。ghost が尽きたら grand は止まる。
- grand は barostat があると止まる。箱は grand の例では NPT の最後のフレーム(「平均」と書いた出典は無い)。cuFEP は平衡化の
  後半の平均の箱に固定する(独自、計画のとおり)。
- 挿入した水の速度: grand は位置だけを置く(ghost のときの速度のまま)。cuFEP も同じ。
- grand は TIP3P だけ(`assert ff.lower() == "tip3p"`)。4 サイトの水、膜、消去の途中のリガンド、ハミルトニアンの違う
  レプリカの交換は、読めた出典には無い(Ross 2017 の交換は B だけが違うレプリカ)。これらは cuFEP 独自として記録した。
- grand の μ'ex と V° の既定値(-6.09 kcal/mol、30.345 Å³)は grand の設定(分散補正なし、Lennard-Jones を 1.0〜1.2 nm で
  切り替え)の値で、コードにも「may need to be changed for specific simulation parameters」とある。OPC の値はどの出典にも無い。

### 2026-11-25: 実装

- `src/cufep/sampling/gcmc.py`(新規): `GcmcSpec`、`gcmc_spec`、`ghosts_off`、`WaterSwitch`、`GcmcMover`。試行のエネルギーは
  `relocation_mc.py` の stand-in(`active` で ghost を除く、`water_energies` を追加)。
- 各レプリカの水の一覧: `SharedArrays.won`、`ReplicaWorker.run`(MD の前に Context に入れ、移動のあと書き戻す)、checkpoint
  (`water_on`)、`start_configs.npz`、precalculation の窓の配置(`alchemy/schedule.py`)。
- `workflow.py`: `leg_system`(準備した系から ghost を切った系と移動の仕様)、`alchemical_box` と `positions_in_cell`
  (平衡化の後半の平均の箱に固定)、`leg_spec`(barostat なし)、`samples_settings` と `equil_made_with`(記録)、
  `pressure_of_samples`、`gcmc_numbers`。`alchemy/lrc.py`: tail correction は各配置の ghost を除く。
- 設定 `sampling.gcmc`(既定 off): `interval_ps` 4、`attempts` 10000、`batch_size` 1000、`radius_margin_nm` 0.3、
  `ghost_waters` auto、`excess_chemical_potential_kcal_per_mol` auto、`standard_volume_nm3` auto。
- テスト: unit(`tests/unit/test_gcmc.py`)と GPU(`tests/gpu/test_gpu.py` の末尾)。相互作用の無い水で球の中の水の数が
  Poisson 分布(平均 e^B、井戸があれば e^(B+1))になること、切り替えた Context がそう作った系と同じエネルギーと力を出すこと、
  stand-in が ghost を除くこと、レグを通した記録と再開、precalculation の窓、設定の拒否、固定した箱。
- 別件: unit テストの共有の水の系(`tests/conftest.py` の `water_ligand_system`)が実行ごとに違う配置だった(CPU の
  スレッド)。relocation のテスト 1 本がそれで 1 回落ちたので、1 スレッドで決定的な力にした。

### 2026-11-25: μ'ex と V°(`spikes/m34_water_chemical_potential.py`)と、バルクの水での確認(`spikes/m35_bulk_water_gcmc.py`)

cuFEP の設定(PME、カットオフ 0.9 nm、分散補正、298.15 K、1 bar)、cuPotential、2 回ずつ。μ'ex は水 1 個の消去
(25 窓、窓あたり 3 ns、MBAR と tail)、V° は 1 bar の 2 ns の体積(誤差はブロックから):

| 水 | μ'ex (kcal/mol) | V° (Å³) | 公表値 | 使う値 |
|---|---|---|---|---|
| TIP3P(水 1,386 個) | -6.125 ± 0.053、-6.129 ± 0.049 | 30.319 ± 0.009、30.343 ± 0.005(stock OpenMM: 30.343、30.328) | -6.09、30.345(grand) | 公表値(誤差の中で一致) |
| OPC(水 1,384 個) | -8.051 ± 0.068、-8.046 ± 0.074 | 29.995 ± 0.009、30.003 ± 0.006(stock OpenMM: 30.044、30.042) | 無い | -8.05、30.000(cuFEP の測定) |

- μ'ex は tail を含む値。移動で切り替わる水は NonbondedForce の分散補正の中に入るので、その分(TIP3P -0.113、
  OPC -0.164 kcal/mol)を含む値が run と整合する。MBAR だけの値は -6.013 と -7.886。
- 最小の overlap は 0.089〜0.091、独立なサンプルは窓あたり 1,030 以上。
- バルクの水での確認(1 bar の密度の箱、球 0.8 nm、4 ns、4 fs): 球の中の水の数は TIP3P(μ'ex -6.125)70.34 ± 0.26
  (V_球/V° = 70.67)、TIP3P(-6.04)70.63 ± 0.15、OPC(-8.05)71.48 ± 0.17(71.46)、OPC(-7.93)71.75 ± 0.17。箱全体の
  水の数は開始から -0.38 %、-0.18 %、+0.20 %、+0.40 %。箱全体の数は grand canonical ensemble では 0.7 % ほど揺らぎ、
  ゆっくり変わるので、数 ns の run ではこの違いを見分けられない(1 fs と 2 fs の run も -0.5 %〜+0.8 % の中)。
- **μ'ex の誤差が結果に効く大きさ**: 複合体レグの自由エネルギーは、μ'ex の誤差 × (リガンドを消したときに球に増える水の数)
  だけ変わる(溶媒レグは圧力一定で、この項が無い)。0.05 kcal/mol と水 7 個なら 0.35 kcal/mol。いまの μ'ex の標準誤差は
  2 回の平均で TIP3P 0.036、OPC 0.050 kcal/mol。小さくするには水の箱の消去を長くする(1 回きりの計算)。

### 2026-11-25: 通しの確認(A2A 4k、短い設定、開発中のツリー)

準備、ghost を切った平衡化、拘束の MD、precalculation(30 窓、プールのワーカーが移動を行う)、HREX(GPU 4 枚、
ワーカー 16、30 レプリカ)、解析まで通った(窓あたり 0.075 ns なので自由エネルギーの値に意味は無い)。ghost は水 14,709 個の
うち 72 個、球は 8.0 Å、B = -6.011。移動: 挿入 359,928 回中 261、削除 360,072 回中 249。球の中の水の数の平均は、
結合した側の窓で 7 前後、リガンドを消した側の最後の 3 窓で 13.5、13.3、14.2。箱は平衡化の後半の平均(8.4664 ×
8.4664 × 9.8789 nm)に固定され、結果の圧力は 1 bar と記録された。

### 2026-11-25: 実装のレビュー(書いた本人でない読み手、20 分)と対応

P1 は無し(移動が誤った分布を出す入力は作れなかった。受理の因子、ghost の選び方、2 段の判定、バッチと 1 回ずつの試行の
等価、ワーカーでの一覧の受け渡し、固定した箱、tail correction を読んで確認)。

- P2: ghost が尽きたときの停止が、ワーカーの失敗として扱われていた(ワーカーを作り直して別の乱数でやり直し、2 回目は
  バックエンドの失敗として stock OpenMM でのやり直しになる)。→ 移動が続けられないという停止(`RelocationError` と
  `GcmcError`)は、ワーカーが「refused」として返し、sampler は `MovesError` で理由を出して止める。作り直しもやり直しも
  しない。relocation の「箱が球に対して小さくなった」も同じ扱いになった。テスト: unit(inline)と GPU(ワーカープロセス)。
- P2: stand-in のエネルギーが 0 でない系での受理のテストが無かった(分布のテストは相互作用の無い水)。→ 2 本足した。
  (1) 1 回のバッチの各試行の受理が、独立に計算した stand-in のエネルギーで、公表の式どおりであること(決定的)。
  (2) リガンドとも互いにも相互作用する分子で、stand-in を使う mover と、stand-in を使わない mover(毎回シミュレーションの
  エネルギーだけで判定)が同じ水の数を出すこと(4.51 と 4.49、6 本ずつの鎖の標準偏差 0.19 と 0.24)。コードを変えた
  写しで落ちることを確かめた: 2 回目の判定で stand-in を引かない(水 15.0)、削除の符号を逆にする(23.2)、挿入の符号を
  逆にする((1) が落ちる)。
- P3: MC barostat は ghost を分子として数える(72 個、810 nm³ で 3.7 bar)。`docs/protocol.md` に書いた。
- P3: `cufep validate` が ghost を切らない系を作っていた。→ `leg_system` を通す。
- P3(仮説): ghost のサイトがリガンドの原子に重なるとソフトコアの力が NaN になるか。→ GPU で確かめた: 距離 1e-6 nm
  までは両方のバックエンド、全ての窓でエネルギーも力も有限。NaN は距離が厳密に 0 のときだけ。対処なし。

### 2026-11-25: 移動 1 回の試行数の既定と、止まった短い run

- 既定の `attempts` を 2500 にした(loch の例は 10000)。試行が 1 つ通るたびに Context のパラメータを書き換える必要があり、
  4k の複合体レグでは 10000 回の試行で、結合した側の窓で 1 回の移動あたり 1.3 個、リガンドを消した側の窓で 16 個の水が
  出入りした(その窓の球の中の水は 14 個前後)。4 分の 1 でも、結合した側で 3 回に 1 回、消した側で 1 回の移動あたり 4 個が
  出入りする。根拠は `docs/configuration.md` に書いた。配置が変わらないあいだは、試行によらない計算(stand-in への配置の
  転送、球の中の水、削除の候補のエネルギー)を使い回し、通らなかった試行を Context から戻す書き換えも次の試行と
  まとめるようにした(結果は変えない)。
- 短い確認の run(GPU 8 枚、ワーカー 31、MPS)5 回のうち 1 回が、反復 84 で「CUDA_ERROR_LAUNCH_FAILED」になり、その GPU の
  MPS が落ちてレグが失敗した。同じ設定をもう一度流すと最後まで走った。1 プロセスで同じレグの移動を 450 回ずつ 3 通り
  行っても起きなかった。原因は分かっていない。本番の run で同じ失敗が出たら、そのレグは checkpoint から続けられる。

### 2026-11-25: A2A、GCMC、1 本ずつ(イメージ `cufep:8252ccd`、GPU 8 枚)

- 設定: セットの設定(5 ns、1 計算)に `sampling.gcmc.enabled: true` だけを足したもの。OPC の run はそれに
  `forcefield.water: amber14/opc.xml`。`runs/configs/a2a_gcmc_{tip3p,opc}_8252ccd.yaml`、出力
  `runs/benchmark_a2a_full_gcmc_{tip3p,opc}_8252ccd`。OPC の設定と、移動なしの二重層 + OPC の run(0e3b2a9)の設定との違いは、
  その後に増えた設定の既定値、`sampling.gcmc.enabled`、出力先だけ。
- 順番: 4k(TIP3P)、4k(OPC)、4e(TIP3P)、4e(OPC)。残りをどちらの水で流すかはオーナーが決める。

結果(kcal/mol、1 リガンド 1 計算)。「GCMC なし」は TIP3P では e77a6c5、OPC では 0e3b2a9 の run:

| リガンド | 水 | 実験値 | GCMC あり(この run) | GCMC なし | 複合体レグ あり / なし | 溶媒レグ あり / なし | 拘束 | 窓 複合体 / 溶媒 | 最小の overlap | 独立なサンプルが 50 未満の窓 | 挿入 / 削除(行った数 / 試行) | 球の中の水(最初の窓 → 最後の窓の平均) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4k | TIP3P | -11.54 | -17.32 ± 0.42 | -20.32 ± 0.52 | 41.78 ± 0.41 / 44.87 | 14.86 ± 0.11 / 14.97 | ASN 253 の水素結合 (800 フレーム中 762) | 31 / 20 | 0.154 / 0.182 | 複合体レグ (最少 44) | 15,033 / 48,438,948、14,907 / 48,436,052 | 8.7 → 20.3 |
| 4k | OPC | -11.54 | -19.96 ± 0.29 | -19.15 ± 0.35 | 44.97 ± 0.26 / 43.89 | 15.45 ± 0.13 / 15.16 | ASN 253 の水素結合 (800 フレーム中 790) | 30 / 17 | 0.154 / 0.196 | 複合体レグ (最少 33) | 4,900 / 46,880,428、4,882 / 46,869,572 | 8.7 → 16.4 |
| 4e | TIP3P | -12.07 | -17.21 ± 0.36 | -13.50 ± 0.34 | 43.27 ± 0.34 / 39.76 | 16.47 ± 0.12 / 16.54 | ASN 253 の水素結合 (800 フレーム中 787)。GCMC なしの run は PHE 168 とのスタッキング | 32 / 19 | 0.160 / 0.197 | 複合体レグ (最少 49) | 19,911 / 49,991,480、19,792 / 50,008,520 | 8.9 → 20.8 |
| 4e | OPC | -12.07 | -18.26 ± 0.54 | -20.61 ± 0.47 | 45.10 ± 0.51 / 47.34 | 17.00 ± 0.18 / 16.98 | PHE 168 とのスタッキング (800 フレーム中 798) | 31 / 18 | 0.167 / 0.179 | 複合体レグ (最少 21) | 5,539 / 48,437,677、5,477 / 48,437,323 | 8.1 → 17.8 |

- 4k(TIP3P): 二重層は POPC 91 + 91、平衡化の最後の 2 つの四半期のあいだの面積の変化は -1.64 %。ghost は水 14,703 個の
  うち 72 個、球は 8.0 Å、B = -6.011。両方のレグが cuPotential で走り、ワーカーの作り直しは無い。複合体レグの往復は 0 回。
  球の中の水の数の平均は、窓ごとに 8.7、8.6、…、7.3(中ほど)、…、12.8、16.7、18.4、19.5、20.3(リガンドを消した側)。
  TIP3P で移動なしの 4k には、ほかに -14.49、-13.07、-16.50 の 3 本(複合体レグ 38.78、37.36、40.90)、RelocationMC ありの
  -16.61 ± 0.37(複合体レグ 41.00)がある(計画 2026-11-22)。
- 4k(OPC): 二重層は POPC 91 + 91、面積の変化は -0.96 %。ghost は水 14,705 個のうち 73 個、B = -9.308。両方のレグが
  cuPotential で走り、ワーカーの作り直しは無い。複合体レグの往復は 0 回。球の中の水の数の平均は 8.7(結合した側)、
  7.5(中ほど)、12.8、14.8、15.3、16.1、16.4(リガンドを消した側)。OPC で RelocationMC ありの run(9989cdb)は
  -19.16 ± 0.59(複合体レグ 43.86)。
- 4e(TIP3P): 二重層は POPC 90 + 90、面積の変化は -0.54 %。ghost は水 14,707 個のうち 67 個、球は 7.8 Å、B = -6.089。
  両方のレグが cuPotential で走り、ワーカーの作り直しは無い。複合体レグの往復は 1 回。球の中の水の数の平均は 8.9
  (結合した側)、6.3(中ほど)、16.7、18.4、19.1、20.5、20.8(リガンドを消した側)。拘束は ASN 253 の水素結合で、
  e77a6c5 の run(PHE 168 とのスタッキング)と違う。TIP3P で移動なしの 4e には、`interactions` で拘束が ASN 253 だった run
  -16.85 ± 0.64(複合体レグ 43.04、fce6929)、ALA 59 だった -12.14 ± 0.49(38.71)、
  GLU 169 だった -15.47 ± 1.00(41.83)がある(`PROGRESS.md` の A2A の例の行)。
- 4e(OPC): 二重層は POPC 91 + 91、面積の変化は -0.33 %。ghost は水 14,705 個のうち 67 個、B = -9.386。両方のレグが
  cuPotential で走り、ワーカーの作り直しは無い。複合体レグの往復は 0 回。球の中の水の数の平均は 8.1(結合した側)、
  6.3(中ほど)、13.0、15.7、17.3、17.8(リガンドを消した側)。拘束は GCMC なしの run(0e3b2a9)と同じ PHE 168。

4 本の実験値との差(kcal/mol)。「同じ拘束の GCMC なし」は、拘束が同じ残基だった run:

| リガンド、水 | GCMC あり | 同じ拘束の GCMC なし | 差(あり - なし) |
|---|---|---|---|
| 4k、TIP3P | -5.78 | -8.78(e77a6c5。ほかの 3 本は -2.95、-1.53、-4.96) | +3.00(ほかの 3 本の平均とは -2.63) |
| 4k、OPC | -8.42 | -7.61(0e3b2a9) | -0.81 |
| 4e、TIP3P | -5.14 | -4.78(fce6929。e77a6c5 は拘束が PHE 168 で -1.43) | -0.36 |
| 4e、OPC | -6.19 | -8.54(0e3b2a9) | +2.35 |

- 4 本とも GCMC の移動は行われ(TIP3P で挿入・削除が 1.5〜2 万回ずつ、OPC で約 5 千回ずつ)、リガンドを消した側の窓で球の中の水が
  7〜9 個から 16〜21 個に増えた。GPU のエラーとワーカーの作り直しは 4 本とも無い。
- 4 本とも結合が実験値より 5〜8 kcal/mol 強い。GCMC のあり・なしの差は +3.0、-0.8、-0.4、+2.4 で、向きがそろっていない
  (1 計算のばらつきは 4k で標準偏差 1.7)。残りをどちらの水で流すかはオーナーの判断を待つ。

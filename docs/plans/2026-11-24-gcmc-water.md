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

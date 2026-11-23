# 二重層は拘束条件の MD だけ、複合体レグの FEP は水中
Status: In progress (2026-11-24: オーナーの指示で A2A の run を 4k と 4e の 2 本で止めた。設定と PR #14 の扱いはオーナーの決定待ち)
Refs: this pull request; `docs/configuration.md` (`membrane.complex_leg`); `docs/protocol.md` ("Systems", "Steps of one leg"); `benchmarks/results/a2a/`

## Context

オーナーの依頼(2026-11-23): 二重層の中で FEP を実行することが A2A の精度を悪くしているかを確かめる。
拘束条件を決める MD は二重層の中で行う。FEP 本体だけを二重層なしで実行して、結果が変わるかを
A2A のベンチマークで見る。

## Approach

- 設定 `membrane.complex_leg: bilayer | water`。既定は `bilayer`(今の動作)。
- `water` のとき:
  - 拘束条件: 今と同じく二重層の中(平衡化、`restraints.selection: interactions` の MD、Boresch)。
  - 複合体レグの alchemical run: 水中の系。入力のポーズを水の箱に入れ、可溶性タンパク質の複合体と
    同じ手順で実行する。
  - 溶媒レグ: 変更なし。
- 準備で複合体を 2 つ作る。`prep/complex.*` が水中(alchemical run の系)、
  `prep/complex_bilayer.*` が二重層の中(拘束条件の MD の系)。同じ入力のポーズから作る。
  レグ自身のファイル(`prep/complex.*`、`equil.npz`)は alchemical run がサンプルする系のままなので、
  拘束条件より後ろは可溶性の複合体と同じ経路を通る。
- アンカー 6 原子は、二重層の系で選び、同じ原子として水中の系に写す。2 つの系の溶質(タンパク質と
  リガンド)が同じ原子の並びでなければ、準備は何も書かずに止まる。候補は、alchemical run が始まる
  配置(水中の平衡化の終わり)でもアンカーの規則を満たすものだけを取る。
- `water` は `membrane.enabled: true` と `restraints.selection: interactions` の組み合わせでだけ使える。
  それ以外は設定を読むときに拒否する。
- 結果には、拘束条件を二重層の中で選び、alchemical run を水中で実行したことが出る。

## Invariants

- 新しい判定は足さない。使えない組み合わせの拒否は設定どうしの矛盾で、結果への判定ではない。
- `membrane.complex_leg` は cuFEP 独自の設定で、出典はない。`docs/configuration.md` にそう書く。
  既定値は今の動作のままで、オーナーが決めるまで変えない。
- 自動の動作は足さない。二重層を作るかどうかも、どの段階で使うかも設定が決める。
- 既定の解析の数値は変えない(`ANALYSIS_VERSION` 10、`tests/unit/data/analysis_golden.json`)。
  保存済みの結果と、既定の設定で作ったディレクトリはそのまま使える。
- 1 リガンド 1 計算。cuPotential プラグインの速度や実行時間は書かない。

## Verification

- CPU: `scripts/in_docker.sh python -m pytest -q`。設定の拒否、2 つの系のあいだで写した拘束の
  エネルギー、拘束条件の段階の再開、別の設定で作ったディレクトリの拒否、`cufep run` と
  `cufep analyze` が同じ結果を出すこと。
- GPU: `scripts/in_docker.sh --gpus 0 python -m pytest tests/gpu -q`。短い設定で A2A の例を 1 回。
- Runs: A2A のセットの設定に `membrane.complex_leg: water` を足し、コミットから作ったイメージで、
  1 リガンドずつ全 GPU で実行する。順番は 4k、4e、残り 15 本。1 本ごとに報告する。
- 比較対象(オーナーの決定): `benchmarks/results/a2a/` の全体の run(コミット e77a6c5)。この run で
  複合体レグの窓を 1 つずつ順に始めた 7 本(4c、4h、4j、4k、4n、4q、4r)は、差に窓の始め方の違いも
  入るので、表で印を付ける。
- 出すもの: リガンドごとの結合自由エネルギーと標準誤差、実験値、比較対象との差、各レグ、選ばれた
  拘束、窓の数、overlap、独立なサンプルの数、タンパク質の C-alpha の RMSD。セットでは、差の平均と
  その標準誤差、平均符号付き誤差、平均を除いた RMSE、Kendall tau。数値で出し、cuFEP 自身の判定は
  書かない。

## 記録

### 2026-11-23: 実装

- `membrane.complex_leg`(既定 `bilayer`)を実装した。準備(`prepare_ligand`)が 2 つの複合体を書き、
  拘束条件の段階(`restraint_from_the_bilayer`)が二重層の系を平衡化して(`restraint_equil.npz`)
  拘束を選び、アンカーを水中の系の番号に写す(`solute_correspondence`、`restraint_in_system`)。
  `restraint.json` は、選んだときの系と番号(`md_system`)と、alchemical run が始まる配置
  (`alchemical_start`)を記録し、レグはその 2 つのファイルと照合する(`check_restraint_of_leg`)。
- 書いていない者による差分のレビュー(1 回): 誤った数値になる入力やディレクトリの状態は
  見つからなかった。指摘 5 件のうち 4 件を直した。二重層の平衡化のファイルをレグ自身も設定と
  照合する(`check_restraint_equil`)。原子の対応に鎖の名前を含める。`docs/protocol.md` の溶媒レグの
  箱の形と `ligand_rmsd_from_input_A` の記述。変えなかった 1 件: `membrane.enabled: false` のときの
  `complex_leg` は、ほかの `membrane.*` と同じく何もしない。
- 溶媒レグ: cuPotential プラグインでは変更なし。stock OpenMM では `solvation.box_shape: auto` が
  水中の複合体の形(dodecahedron)を溶媒レグの箱にも与える。
- 単体テスト 714 passed、GPU テスト 26 passed(イメージ `cufep:dev`、作業ツリーのソース)。
  実物の二重層を作る準備は単体テストでは通していない(二重層の構築を置き換えたテストだけ)。
  設定を使った run はまだ無い。

### 2026-11-23: 短い設定での確認(イメージ `cufep:a34bbe2`、4k、全 GPU)

- `runs/a2a_water_smoke_a34bbe2`(二重層の平衡化 0.5 ns、拘束条件の MD 4 × 0.3 ns、窓あたりの
  production 0.1 ns)。終了コード 0。数値は短すぎて結果ではない。
- 準備が書いた系: `prep/complex.*` は 61,452 原子で脂質が無く(水 18,883 分子)、等方圧、
  dodecahedron。`prep/complex_bilayer.*` は 73,275 原子で POPC 91 + 91。アンカーは 2 つの系で同じ番号
  (この入力には結晶水が無く、リガンドの前の原子は消えない)。
- レグの順番: 水中の平衡化、二重層の平衡化、拘束条件の MD 4 本(cuPotential)、拘束(ASN 253 との
  水素結合)、水中の系で 30 窓の alchemical run。`restraint.json` に `md_system`、`start`、
  `alchemical_start`。結果の `restraint_choice.md_system` は `complex_bilayer`、`bilayer` は二重層の
  平衡化の数値。
- 同じコマンドをもう一度: 何も計算し直さず、リガンドと両レグの `result.json`、`restraint.json`、
  `equil.npz`、`restraint_equil.npz` は変わらない(sha256 が同じ)。`cufep analyze` は複合体レグの
  結果と同じ値を出す(209.90 ± 2.68 kJ/mol)。

### 2026-11-24: A2A、1 リガンドずつ(イメージ `cufep:a34bbe2`、全 GPU)

`runs/benchmark_a2a_full_water_a34bbe2`。設定は `runs/benchmark_a2a_full_41760a1/resolved_config.yaml`
(セットの設定、窓あたりの production 5 ns、1 計算)に `membrane.complex_leg: water` を足し、
`solvation.box_shape` を `auto` にしたもの。比較対象は `runs/benchmark_a2a_full_e77a6c5`
(`benchmarks/results/a2a/`)。kcal/mol。※ は、比較対象が複合体レグの窓を 1 つずつ順に始めた
リガンド(差に窓の始め方の違いも入る)。「拘束」は選ばれた相互作用の残基と、800 フレームのうち
それがあったフレームの数。RMSD は production の後半の、タンパク質の C-alpha の値(A)。

| リガンド | 実験 | 水中 | 二重層 | 差 | 複合体レグ 水中 / 二重層 | 溶媒レグ 水中 / 二重層 | 拘束 水中 / 二重層 | 窓 複合体 / 溶媒 | 最小 overlap | 独立なサンプルの最小 | RMSD 水中 / 二重層 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 4k ※ | -11.54 | -12.11 ± 0.53 | -20.32 ± 0.52 | +8.21 | 36.60 / 44.87 | 14.90 / 14.97 | ASN 253 (784) / ASN 253 (783) | 32 / 19 | 0.154 / 0.171 | 12 / 193 | 2.25 / 1.13 |
| 4e | -12.07 | -16.17 ± 0.47 | -13.50 ± 0.34 | -2.66 | 42.59 / 39.76 | 16.60 / 16.54 | PHE 168 (794) / PHE 168 (785) | 32 / 18 | 0.166 / 0.169 | 11 / 243 | 1.75 / 1.09 |

- 4k: 窓を walk で始めた二重層の 3 本(`runs/acc_base_a2a_4k_{1,2,3}`、イメージ e77a6c5)は
  -14.49、-13.07、-16.50(平均 -14.69、標準偏差 1.72)。複合体レグ 38.78、37.36、40.90。
  C-alpha の RMSD 1.60、1.60、1.46 A。拘束の解放は水中 -9.58、二重層 -9.57 から -9.60。

### 2026-11-24: オーナーの指示で run を止めた

- 3 本目(4g)の途中、2026-11-23T16:06:55Z に止めた。4g は準備と水中の平衡化までで、結果は無い。
  結果があるのは上の表の 4k と 4e の 2 本。セットの統計は出していない。
- 2 本の差(水中 - 二重層): 4k は比較対象(窓を 1 つずつ順に始めた run)に対して +8.21、窓を walk で
  始めた二重層の 3 本の平均に対して +2.58。4e は -2.66。1 計算のばらつきとして測ってあるのは 4k の
  二重層の 3 本の標準偏差 1.72。
- ディレクトリは残してある(`runs/benchmark_a2a_full_water_a34bbe2`、`runs/a2a_water_smoke_a34bbe2`、
  進行の記録 `runs/a2a_water_a34bbe2.log`。追跡していない)。設定は PR #14 にあり、既定は `bilayer` のまま。

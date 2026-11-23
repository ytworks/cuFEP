# 二重層は拘束条件の MD だけ、複合体レグの FEP は水中
Status: In progress
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

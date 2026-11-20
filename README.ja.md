# cuFEP

[![CI](https://github.com/ytworks/cuFEP/actions/workflows/ci.yml/badge.svg)](https://github.com/ytworks/cuFEP/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)
![OpenMM 8.4](https://img.shields.io/badge/OpenMM-8.4-blue.svg)

[English](README.md) | **日本語**

cuFEP は、低分子のタンパク質への**絶対結合自由エネルギー (ABFE)** を
[OpenMM](https://openmm.org) で計算するソフトウェアです。タンパク質の PDB と、リガンドをポケットに
置いた SDF を渡すと、系の準備、restraint と lambda window の選択、REST2 付きの Hamiltonian replica
exchange をマシンの全 GPU で行い、結合自由エネルギーを標準誤差と、alchemical 計算のベストプラクティス
文献に基づく検査の結果とともに報告します。計算のバックエンドは既定で NVIDIA の
[cuPotential](https://github.com/NVIDIA/cuPotential) (OpenMM のプラグイン) で、stock OpenMM を
2 つ目のバックエンドとしていつでも使えます。

> **状態: バージョン 0.1、開発中。** 中性のリガンドのみ対応し、補因子にはまだ対応していません。
> インターフェースと既定値は今後変わることがあります。現在の状態は [PROGRESS.md](PROGRESS.md)
> (英語)、制限は [制限](#制限) を参照してください。

- [特長](#特長)
- [仕組み](#仕組み)
- [必要なもの](#必要なもの)
- [インストール](#インストール)
- [クイックスタート](#クイックスタート)
- [出力](#出力)
- [結果の読み方](#結果の読み方)
- [設定](#設定)
- [例](#例)
- [ベンチマーク](#ベンチマーク)
- [制限](#制限)
- [ドキュメント](#ドキュメント)
- [開発](#開発)
- [参考文献](#参考文献)
- [ライセンス](#ライセンス)

## 特長

- **入力は 2 つのファイル。** タンパク質の PDB とリガンドの SDF です。SDF に複数の分子があれば、
  分子ごとに 1 つの計算になります。それ以外の設定にはすべて根拠を示した推奨の既定値があり、YAML
  で変えられます。
- **水溶性タンパク質と膜タンパク質。** `membrane: {enabled: true}` で、複合体の向きを決めて POPC の
  脂質二重膜に埋め込みます。膜を付けるかどうかを cuFEP が自分で判断することはありません。
- **確立したプロトコル。** 自動で選ぶ Boresch restraints による double decoupling、リガンドと
  ポケットに REST2 をかけた Hamiltonian replica exchange、MBAR。既定値と用語は確立した方法と
  文献に合わせ、各既定値の根拠を文書に書いています。
- **lambda window。** 既定は adaptive lambda scheduling (Midgley ら 2025)。leg と stage ごとに
  window の数や lambda の値の列で固定することもできます。
- **判定は文献の基準だけ。** 隣り合う lambda window の overlap (0.03 以上) と、window ごとの相関の
  ない標本の数 (50 以上) です。それ以外 (forward と reverse の系列、前半と後半、ブロック、BAR) は
  判定なしの数値として報告します。
- **全 GPU を使う。** 1 つの計算の replica を、仕事量に比例して空いている全 GPU に配り、NVIDIA MPS で
  GPU 1 枚に複数載せます。GPU 1 枚でも動きます。
- **再開できる。** 計算は止めて続きから再開できます。1 つの leg だけを、長くしたり lambda window を
  増やしたりして計算し直すことができ、残りの結果はそのまま使われます。
- **ベンチマーク。** Schrödinger の公開結合自由エネルギーベンチマークのキナーゼ (TYK2) と、脂質
  二重膜中の GPCR (A2A) で、実験値と、そこで公開されている FEP+ の予測値に対して比較します。

## 仕組み

リガンドごとに、2 つの leg から結合自由エネルギーを求めます。タンパク質から切り離す complex leg と、
水から切り離す solvent leg です。

1. **準備。** タンパク質は PDB のまま準備します。ファイルにある水素は残し、欠けた重原子は補い、
   鎖の切れ目にはキャップを付け、欠けたループはモデリングしません。リガンドのパラメータは
   OpenFF 2.2.1 と AM1-BCC 電荷です。系は水で溶媒和するか、二重膜に埋め込みます
   ([protocol: Inputs](docs/protocol.md#inputs)、[Systems](docs/protocol.md#systems))。
2. **pre-equilibration。** 相互作用がすべてある系で、最小化、NVT、NPT のあと、complex leg では 5 ns
   の MD を行います。
3. **Boresch restraint** をその軌跡から OpenFE の規則で選びます
   ([Steps of one leg](docs/protocol.md#steps-of-one-leg))。
4. **lambda window** を adaptive lambda scheduling で決めます。短い前計算で、隣り合う window の
   overlap が足りるまで window を差し込みます
   ([Adaptive lambda scheduling](docs/protocol.md#adaptive-lambda-scheduling))。
5. **Hamiltonian replica exchange。** lambda window ごとに 1 つの replica で、リガンドとそこから
   0.4 nm 以内の残基に REST2 をかけます ([REST2](docs/protocol.md#rest2))。window ごとに 1 ns の
   equilibration と 10 ns の production です。
6. **MBAR。** window ごとに相関を除いた標本で計算し、標準誤差は時間ブロックから求め、文献の検査を
   行います ([Checks of the analysis](docs/protocol.md#checks-of-the-analysis))。

## 必要なもの

- Docker と
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
  が使える Linux。
- NVIDIA の GPU。イメージは CUDA 13.4 向けです。NVIDIA RTX PRO 6000 Blackwell Server Edition と
  ドライバ 595.99.02 で確認しており、このドライバではイメージ内の forward-compatibility パッケージで
  CUDA 13.4 が動きます。他のドライバや GPU で動くかどうかは NVIDIA の
  [CUDA 互換性の規則](https://docs.nvidia.com/deploy/cuda-compatibility/latest/forward-compatibility.html)
  に従います。
- イメージと計算結果を置くディスク容量。

それ以外 (OpenMM 8.4、cuPotential プラグイン、OpenFF、AmberTools、pymbar、PDBFixer、RDKit) は
すべて Docker イメージに入っています。

## インストール

このリポジトリからイメージを作ります。

```bash
git clone https://github.com/ytworks/cuFEP.git
cd cuFEP
docker build -f docker/Dockerfile -t cufep:latest .
```

ビルドの際に NVIDIA の cuPotential ランタイムをダウンロードします。これは NVIDIA のプロプライエタリ
ソフトウェアです。これを含むイメージを公開する前に [docs/LICENSING.md](docs/LICENSING.md) を
読んでください。NVIDIA のプロプライエタリソフトウェアを含まない stock OpenMM だけのイメージは
`--build-arg WITH_CUPOTENTIAL=0` で作れます。

```bash
docker build -f docker/Dockerfile --build-arg WITH_CUPOTENTIAL=0 -t cufep:stock .
```

ビルド済みのイメージはまだ公開していません。

## クイックスタート

```bash
# 以下のコマンドのためのシェル関数
cufep() { docker run --rm --gpus all --shm-size=16g -u "$(id -u):$(id -g)" \
                     -v "$PWD":/work cufep:latest "$@"; }

cufep check-env                                   # ソフトウェア、GPU、両バックエンドの確認
cufep init protein.pdb ligand.sdf -o config.yaml  # 最小の設定ファイルを書く
cufep plan config.yaml                            # 計算せずに、何を行うかを表示
cufep run config.yaml                             # 実行、または再開
```

`--shm-size=16g` (または `--ipc=host`) は重要です。replica の座標はワーカープロセスの間でメモリを
共有して受け渡しており、Docker の既定の 64 MB では足りません。指定がないと、cuFEP は出力
ディレクトリのファイルで代用し、そのことをログに書きます。

`cufep run` はリガンドごとに、時刻に続けて次のような 1 行を表示します。

```text
RESULT ejm_31: dG_bind = ... ± ... kcal/mol (standard error from time blocks, 1 repeat)
```

## 出力

| ファイル | 内容 |
|---|---|
| `resolved_config.yaml`、`auto_report.md` | 使ったすべての設定と、計算で決めた値の理由 |
| `protein_prepared.pdb` | 計算に使ったタンパク質 |
| `summary.csv`、`summary.json` | 結合自由エネルギー、不確かさ、メッセージ |
| `timing.json` | 段階とジョブごとの時間、GPU の使用率 |
| `<ligand>/repeat_<n>/<leg>/` | leg の restraint、lambda window (`schedule.yaml`)、標本、結果 (`result.json`)、ログ |
| `<ligand>/repeat_<n>/analysis.png` | 両 leg の forward と reverse の系列、production の前半と後半とブロック、overlap matrix |

既定では、リガンドごとに 1 回だけ計算します。不確かさは、報告する自由
エネルギー (MBAR からテール補正を引いたもの) の標準誤差で、すべての lambda window に共通な
production の時間ブロックから求め、解析的な値より小さくはしません。MBAR 自身の標準誤差も並べて
残します (`mbar_error`)。`sampling.repeats` を使うと計算を独立に複数回行い、そのばらつきを報告します。

## 結果の読み方

解析はすべての leg を検査します
([protocol: Checks of the analysis](docs/protocol.md#checks-of-the-analysis))。基準は文献のもの
(Klimovich、Shirts、Mobley 2015、Mey ら 2020) です。cuFEP が自動で計算し直すことはありません。
該当する leg を計算し直し、計算の残りは結果をそのまま使います。

| メッセージ | 意味 | 対処 |
|---|---|---|
| `NOT RELIABLE` | production で、隣り合う 2 つの lambda window の overlap が、全 iteration で見たとき、または自由エネルギーを推定する標本で見たときに 0.03 未満。overlap がなければ自由エネルギーは使えない | その stage の lambda window を増やして leg を最初から計算し直す。推定に使う標本だけが基準を下回る場合は、production を長くすることも有効 |
| `few independent samples` | ある lambda window が MBAR に渡す相関のない標本が 50 未満 (Mey ら 2020)。自由エネルギーとその不確かさが少ない標本に基づいている | leg を長くする。止まったところから続く |
| `FAILED` | leg が終わらなかった、中断された、またはその標本では自由エネルギーが決まらない。そのリガンドはその leg からの結合自由エネルギーを持たない | メッセージを読み、leg を計算し直す |
| `LEGS MADE UNDER DIFFERENT CONDITIONS` | complex leg と solvent leg (または 2 つの repeat) で、温度、圧力、準備した系、テール補正が異なる。そのリガンドは結合自由エネルギーを持たない | 異なる leg を、他の leg の条件で計算し直す |
| `old analysis` | この実行で標本を取らなかった leg が以前の版の解析結果を持ち、その標本が残っていない | その leg を計算し直す |
| `the standard error of this leg is the analytical one ...` | leg が短く時間ブロックが取れないため、標準誤差は標本を独立とみなした値 | leg を長くする |
| `parts of the production without an estimate` | forward と reverse の系列の一部の点やブロックで、標本が少ないなどの理由で自由エネルギーが決まらない。それらは `analysis.png` から除かれる | 何もしなくてよい |
| `parts of the production whose lambda windows overlap below the level of a reliable estimate` | 系列、前半と後半、ブロックの一部が、0.03 未満の overlap に基づいている。それらの推定はこの注記付きで `analysis.png` に示される | 何もしなくてよい |
| `checks of this version not made` | `cufep benchmark report` が以前の版の cuFEP が書いた要約を読んだ | leg を解析し直す |
| `origin of files not recorded` | leg が、何から作ったかの記録のない以前の版のファイルを使った | 必要なら新しい出力ディレクトリにする |

これらが結果に付くメッセージです。解析を判定するのは、文献の 2 つの検査による `NOT RELIABLE` と
`few independent samples` だけです。他のメッセージは、何かが欠けている、比べられない、または
production の一部にしか基づかないことを示します。その他の量は数値として報告します: 平衡化の終わりの膜の変化 (`bilayer`)、
入力ポーズからのリガンドの距離 (`ligand_rmsd_from_input_A`)、結合状態でのリガンドの動き
(`structure`)、forward と reverse の系列、前半と後半、ブロック、BAR。1 回の計算が収束していることを
示せる検査はありません。計算の間に起きなかった遅い変化は、その計算からは見えません。

### leg の再計算

```bash
# 長くする: 1 つのリガンドの complex leg を、lambda window あたり 10 ns から 20 ns に延ばす
# (--production-ns は追加分ではなく全体の長さ)
cufep run config.yaml --ligand ejm_31 --leg complex --production-ns 20

# lambda window を増やす: config.yaml の lambda_schedule を変え (例えば
# adaptive.initial_windows.complex.vanish: 13)、その leg を最初からやり直す
cufep run config.yaml --ligand ejm_31 --leg complex --reschedule

# 標本を取っている最中の leg を見る (最後のチェックポイントまで)
cufep analyze runs/my_ligand/ejm_31/repeat_0/complex
```

`--reschedule` では、leg が持っていた window と標本はその leg のディレクトリの `previous_<n>/` に
移され、pre-equilibration と restraint はそのまま使われます。計算し直した leg は、以前の結果を
`result.previous.json` として残します。leg は、そのファイルを作ったときと同じ条件と入力でのみ
続きを計算します。温度、入力ファイル、準備の設定が異なる実行は止まり、新しい `output.dir` を
求めます
([protocol: Running again in the same output directory](docs/protocol.md#running-again-in-the-same-output-directory))。

## 設定

最小の設定ファイル:

```yaml
system:
  protein: protein.pdb
  ligand: ligand.sdf        # SDF に複数の分子 = それぞれ 1 つの計算
```

`cufep init --full` はすべての設定を既定値付きで書き出します。各設定と既定値の根拠は
[docs/configuration.md](docs/configuration.md) (英語) にあります。よく変える設定:

```yaml
membrane: {enabled: false}           # true でタンパク質の周りに POPC 二重膜を作る
backend: {name: cupotential}         # または openmm (stock)。必要なら openmm に退避する
sampling:
  production_ns: 10                  # lambda window あたり。{complex: 20, solvent: 10} とも書ける
  repeats: 1                         # 計算全体の独立な repeat の数
lambda_schedule:
  mode: adaptive                     # adaptive lambda scheduling。固定にするなら fixed:
  windows:                           # stage ごとに window の数、または lambda の値の列
    complex: {restrain: 6, discharge: 11, vanish: 15}
    solvent: {discharge: 5, vanish: [0.0, 0.12, 0.24, 0.36, 0.48, 0.6, 0.7, 0.77, 0.85, 1.0]}
resources:
  gpus: auto                         # またはデバイス番号のリスト。例 [0, 1]
  oversubscription_factor: auto      # 1 枚の GPU を同時に使うワーカーの数 (NVIDIA MPS)
output: {dir: runs/my_ligand}
```

`cufep plan config.yaml` は実行で使う設定を表示します。fixed では lambda window を、adaptive では
規則と、前計算が始める window を表示します (window そのものは実行の開始時に決まります)。実行は
使ったすべての設定を `resolved_config.yaml` に書きます。

## 例

| ディレクトリ | 系 | 実験値 (kcal/mol) |
|---|---|---|
| [examples/kinase_tyk2](examples/kinase_tyk2) | TYK2 キナーゼ (PDB 4GIH) とリガンド ejm_31、水中 | -9.54 |
| [examples/membrane_a2a](examples/membrane_a2a) | アデノシン A2A 受容体 (PDB 5OLZ) と化合物 4e、POPC 二重膜中 | -12.07 |

```bash
cufep run examples/kinase_tyk2/config.yaml
cufep run examples/membrane_a2a/config.yaml

# パイプラインの短い確認: 1 リガンド、ごく短いサンプリング
cufep benchmark run tyk2 --tier smoke
```

バージョン 0.1 の以前のコミットで、ALS の基準 0.18、ポケットの REST2、lambda window あたり 5 ns の
production (既定は 10 ns) で各 example を 1 回ずつ計算した結果は、TYK2 ejm_31 が
-8.94 ± 0.77 kcal/mol、A2A 4e が -11.95 ± 0.73 kcal/mol でした (1 回の計算の標準誤差。[docs/configuration.md](docs/configuration.md)
の `sampling.rest2.region`)。どちらも 1 回だけの計算で、complex leg を繰り返し計算したときの値の
ばらつきはこれらの誤差より大きくなっています ([protocol: REST2](docs/protocol.md#rest2))。

## ベンチマーク

```bash
cufep benchmark list
cufep benchmark run tyk2 --tier smoke     # 1 リガンド、ごく短い: パイプラインの確認
cufep benchmark run tyk2 --tier subset    # 5 リガンド
cufep benchmark run a2a --tier full       # 全リガンド
cufep benchmark report tyk2 runs/benchmark_tyk2_subset
```

| セット | 標的 | リガンド |
|---|---|---|
| `tyk2` | TYK2 キナーゼ (PDB 4GIH) | 16 |
| `a2a` | アデノシン A2A 受容体 (PDB 5OLZ)、二重膜中 | 17 化合物 (29 状態) |

リガンドごとに 1 回計算します。`--repeats` で回数を増やせます。入力は
[Schrödinger の公開結合自由エネルギーベンチマーク](https://github.com/schrodinger/public_binding_free_energy_benchmark)
(Ross ら 2023、MIT ライセンス。[benchmarks/README.md](benchmarks/README.md)) のものです。レポートは
cuFEP を、実験値と、セットとともに公開されている FEP+ の予測値と比較します。FEP+ の値は相対計算を
実験値の平均にそろえたものなので、レポートは生の誤差 (cuFEP のみ) と、同じ条件での統計を分けて
示します。全セットの結果はまだ公開していません。

## 制限

バージョン 0.1:

- 中性のリガンドのみ (正味の電荷を持つリガンドはメッセージを出して止まる)。補因子は未対応。
- 対称性の補正はしない。リガンドのグラフの対称数は報告する。ポケットの中で裏返る小さな対称な
  リガンドでは、restraint のない window の overlap がなくなり、そのことが結果に出る。
- 対称な POPC 型の二重膜のみ。grand canonical による水のサンプリングはない。
- REST2 はリガンドとそこから 0.4 nm 以内の残基にかかる。それより遠い遅い再配置は、その残基を
  指定したときだけ速くサンプリングされる (`sampling.rest2.residues`)。

[protocol: Limits of version 0.1](docs/protocol.md#limits-of-version-01) を参照してください。

## ドキュメント

| 文書 | 内容 |
|---|---|
| [docs/protocol.md](docs/protocol.md) | 方法: 系、熱力学サイクル、lambda window、REST2、解析と検査、GPU、他のソフトウェアとの比較 (英語) |
| [docs/configuration.md](docs/configuration.md) | すべての設定、既定値とその根拠 (英語) |
| [docs/LICENSING.md](docs/LICENSING.md) | cuFEP、cuPotential ランタイム、ベンチマークの入力のライセンス (英語) |
| [docs/M0_REPORT.md](docs/M0_REPORT.md) | プラグインとプラットフォームの最初のリスク確認 (英語) |
| [docs/plans/](docs/plans/README.md) | 実装計画 (日本語) |
| [PROGRESS.md](PROGRESS.md) | プロジェクトの現在の状態 (英語) |
| [examples/README.md](examples/README.md)、[benchmarks/README.md](benchmarks/README.md) | example とベンチマークのセット (英語) |

## 開発

コントリビュートを歓迎します。[CONTRIBUTING.md](CONTRIBUTING.md) (英語) を参照してください。開発も同じ
Docker イメージの中で、`scripts/in_docker.sh` を通して行います。

```bash
scripts/in_docker.sh python -m pytest -q                     # CPU の単体テスト
scripts/in_docker.sh --gpus 0 python -m pytest tests/gpu -q  # GPU 0 での GPU テスト
```

人にもコーディングエージェントにも共通のプロジェクトの規則は [CLAUDE.md](CLAUDE.md) (Claude Code が
読む。他のエージェントは [AGENTS.md](AGENTS.md) からそこへ導かれる) と
[.claude/rules/](.claude/rules/fep-invariants.md) にあります
(英語)。

## 参考文献

cuFEP は公表された方法を実装しています。使用した場合は、結果の拠りどころとなる方法を引用して
ください。

- Boresch, Tettinger, Leitgeb and Karplus, "Absolute binding free energies: a quantitative
  approach for their calculation", J. Phys. Chem. B 2003, 107, 9535 (Boresch restraints)。
- Wang, Friesner and Berne, "Replica exchange with solute scaling: a more efficient version of
  replica exchange with solute tempering (REST2)", J. Phys. Chem. B 2011, 115, 9431。
- Shirts and Chodera, "Statistically optimal analysis of samples from multiple equilibrium
  states", J. Chem. Phys. 2008, 129, 124105 (MBAR)。
- Midgley, Bariami, Habgood and Mackey, "Adaptive Lambda Scheduling: A Method for Computational
  Efficiency in Free Energy Perturbation Simulations", J. Chem. Inf. Model. 2025, 65, 512。
- Klimovich, Shirts and Mobley, "Guidelines for the analysis of free energy calculations",
  J. Comput. Aided Mol. Des. 2015, 29, 397。
- Mey et al., "Best Practices for Alchemical Free Energy Calculations", Living J. Comput. Mol.
  Sci. 2020, 2, 18378。
- Eastman et al., "OpenMM 8: Molecular Dynamics Simulation with Machine Learning Potentials",
  J. Phys. Chem. B 2024, 128, 109。
- Ross et al., "The maximal and current accuracy of rigorous protein-ligand binding free energy
  calculations", Commun. Chem. 2023, 6, 222 (ベンチマークのセット)。

## ライセンス

cuFEP は [Apache License 2.0](LICENSE) で公開しています。イメージがダウンロードする cuPotential
ランタイムは NVIDIA Software License Agreement の下にある NVIDIA のプロプライエタリソフトウェアです。
この契約は NVIDIA 製品の性能結果の公表も制限しているため、cuFEP の文書にはプラグインの所要時間を
載せていません。ベンチマークの入力は Schrödinger の公開結合自由エネルギーベンチマーク (MIT
ライセンス、`benchmarks/THIRD_PARTY/`) のものです。イメージ内のその他のソフトウェアはそれぞれの
ライセンスに従います。詳細は [docs/LICENSING.md](docs/LICENSING.md) にあります。

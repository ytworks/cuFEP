# cuFEP の精度: 実験値と FEP+ との比較

[English](https://github.com/ytworks/cuFEP/wiki/Accuracy)

このページは、cuFEP の絶対結合自由エネルギーが実験値にどれだけ近いかを、同じベンチマークセットと
ともに公開されている FEP+ の予測値と並べて示します。数値はすべて、各セット全体を 1 回ずつ計算した
結果で、リポジトリの [benchmarks/results](https://github.com/ytworks/cuFEP/tree/main/benchmarks/results)
に記録しています。計算の後で除外、再計算、調整をしたものはありません。解析のチェックを通らなかった
リガンドも、ほかのリガンドと同じように比較し、印を付けています。このページは数値を示すもので、
判定は示しません。

- [要約](#要約)
- [計算したもの](#計算したもの)
- [参照データ](#参照データ)
- [比較の読み方](#比較の読み方)
- [TYK2](#tyk2)
- [A2A](#a2a)
- [別の ABFE 手法 Felis との比較](#別の-abfe-手法-felis-との比較)
- [この結果から言えないこと](#この結果から言えないこと)
- [結果の再現](#結果の再現)
- [出典](#出典)

## 要約

TYK2 (16 リガンド) と A2A (17 化合物) の全セットを、リガンドごとに 1 回、lambda window あたり
5 ns の production で計算しました。下の統計量は 3 つの手法とも Felis 論文の定義で同じように計算した
ものです ([Felis との比較](#別の-abfe-手法-felis-との比較)。kcal/mol、オフセット b =
⟨実験値 − 予測値⟩)。FEP+ の値は公開セットの実験値の平均に全体としてそろえてあるため、そのオフセットは
作り方からほぼ 0 で、2 つの絶対値の手法のオフセットとは比べられません:

| セット | 手法 | R² | Spearman ρ | Kendall τ | オフセット b | 補正後 RMSE |
|---|---|---|---|---|---|---|
| TYK2 (16) | cuFEP、5 ns × 1 回 | 0.19 | 0.21 | 0.12 | 0.62 | 1.74 |
| TYK2 (16) | Felis、5 ns × 3 回 | 0.69 | 0.82 | 0.68 | 0.04 | 0.82 |
| TYK2 (16) | FEP+ | 0.86 | 0.93 | 0.81 | 0.00 | 0.47 |
| A2A (17) | cuFEP、5 ns × 1 回 | 0.15 | 0.48 | 0.38 | 4.12 | 2.26 |
| A2A (17) | Felis、5 ns × 3 回 | 0.55 | 0.81 | 0.65 | 2.74 | 1.16 |
| A2A (17) | FEP+ | 0.50 | 0.73 | 0.59 | 0.16 | 1.08 |

- どちらのセットでも、cuFEP は FEP+ と Felis よりリガンドの順位の当たり方が低く、オフセットを除いた
  誤差も大きくなっています。
- cuFEP の予測は実験値より、TYK2 で平均 0.62 kcal/mol、A2A で 4.12 強く結合する側にずれています。
  Felis は 0.04 と 2.74 です。
- 66 の leg すべてが基準 0.18 の adaptive lambda scheduling で計算され、NOT RELIABLE の結果は
  ありません。TYK2 の 16 リガンド中 12、A2A の 17 化合物中 16 で、ある leg に独立サンプルが 50 未満の
  lambda window があります。
- A2A の example のリガンド 4e を同じ設定で 2 回計算した値は -11.95 と -18.91 kcal/mol でした
  ([A2A](#a2a) を参照)。cuFEP の 1 回の計算は、その標準誤差より大きくばらつきます。

## 計算したもの

cuFEP バージョン 0.1 のコミット `3b45bfe` (pull request #4) を、そのコミットから作った Docker
イメージで、GPU 8 枚の 1 台のマシンで実行しました。計算バックエンドは NVIDIA の OpenMM 用
cuPotential プラグインです。各セットは、その example の設定
([examples/kinase_tyk2/config.yaml](https://github.com/ytworks/cuFEP/blob/main/examples/kinase_tyk2/config.yaml)、
[examples/membrane_a2a/config.yaml](https://github.com/ytworks/cuFEP/blob/main/examples/membrane_a2a/config.yaml))
で、リガンドごとに 1 回計算しました。すべての設定とその根拠は
[docs/configuration.md](https://github.com/ytworks/cuFEP/blob/main/docs/configuration.md)、手法は
[docs/protocol.md](https://github.com/ytworks/cuFEP/blob/main/docs/protocol.md) にあります。

| 設定 | 値 |
|---|---|
| 手法 | double decoupling による絶対結合自由エネルギー。Boresch 拘束、アンカー原子は pre-equilibration から自動で選択 (`restraints.selection: auto`。この計算を行った時点の既定値で、現在の既定値は `interactions`) |
| 力場 | ff14SB (タンパク質)、TIP3P (水)、OpenFF 2.2.1 と AM1-BCC 電荷 (リガンド)、Lipid21 の POPC (A2A の二重膜) |
| 条件 | 298.15 K、1 bar、イオン強度 0.15 M。時間刻み 4 fs、水素質量再分配 (3 amu) |
| アルケミカル計算の前 | complex leg は NVT 0.25 ns、NPT 0.5 ns、MD 5 ns。solvent leg は 0.1、0.2、0.5 ns。A2A は二重膜の平衡化 10 ns |
| lambda window | adaptive lambda scheduling (Midgley ら 2025)、stage の overlap 行列で基準 0.18。complex leg は 3・5・9 window から、solvent leg は 5・9 window から (計算した当時の既定。現在の complex leg は 30 window から始め、隣り合う 2 window だけの overlap を基準 0.30 で読みます) |
| サンプリング | Hamiltonian レプリカ交換、1 iteration 2.5 ps。リガンドと、その 0.4 nm 以内の残基に REST2、scale 1.5 |
| lambda window あたり | 平衡化 1 ns と production 5 ns (cuFEP の既定は 10 ns。example とベンチマークは 5 ns を設定) |
| 解析 | MBAR から長距離分散補正を引いた値。標準誤差は時間ブロックから求め、解析的な値を下回らない (ブロックが作れないほどサンプルが少ないときは解析的な値)。解析のバージョン 10 |

## 参照データ

セットは Schrödinger の公開結合自由エネルギーベンチマーク (Ross ら 2023、MIT ライセンス) のもので、
cuFEP はその入力ファイルを変更せずに使います
([benchmarks/README.md](https://github.com/ytworks/cuFEP/blob/main/benchmarks/README.md))。

| セット | 標的 | リガンド | 実験値の範囲 (kcal/mol) | セットの出典 |
|---|---|---|---|---|
| `tyk2` (`jacs_set/tyk2`) | TYK2 キナーゼ (PDB 4GIH)、水中 | 16 | -11.70 〜 -7.42 (4.28) | doi:10.1021/ja512751q |
| `a2a` (`gpcrs/a2a_hip278`) | アデノシン A2A 受容体 (PDB 5OLZ)、POPC 二重膜中 | 17 化合物 | -12.07 〜 -8.89 (3.19) | doi:10.1021/acs.jcim.0c00449 |

- 実験値: セットとともに与えられた各リガンドの実験の結合自由エネルギー (SDF の `r_exp_dg`。FEP+ の
  結果ファイルの `Exp. dG` と同じ)。
- FEP+: セットとともに公開されている FEP+ の結果ファイルの `Pred. dG` (`tyk2_out.csv`。A2A は
  `a2a_hip278_sbpkacorr_out.csv`)。摂動のマップ上の相対計算から得た値を、公開セットの実験値の平均に
  全体としてそろえたものです。公開された値をそのまま使い、このページのために FEP+ を実行しては
  いません。
- リガンドの状態: TYK2 は化合物ごとに中性のレコードが 1 つです。A2A の元データは 17 化合物に 29 の
  レコード (回転異性体、鏡像異性体、脱プロトン化体) を持ちます。cuFEP 0.1 は中性のリガンドだけを
  扱い、使ったのは FEP+ の結果ファイルが報告するレコードで、どの化合物でも中性です
  ([benchmarks/a2a/ligands.csv](https://github.com/ytworks/cuFEP/blob/main/benchmarks/a2a/ligands.csv))。

## 比較の読み方

- cuFEP は絶対結合自由エネルギーを計算し、FEP+ の値は相対値を実験値の平均にそろえたものです。この
  ため cuFEP の生の平均符号誤差、RMSE、MUE には全値に共通のずれが含まれ、FEP+ のそれはそろえたこと
  で共通のずれが除かれています (公開セット全体では FEP+ の平均符号誤差は 0。ここでのリガンドだけ
  では 0 とは限らない)。この 3 行は同じ条件の比較ではありません。2 つの手法を同じ条件で比べるのは、
  それぞれの平均誤差を除いた RMSE、ペアワイズ ddG RMSE、Kendall τ、Spearman ρ、Pearson r、Pearson R² です。
- かっこ内の値は、リガンドについての 95 % ブートストラップ区間 (再標本 2000 回) です。16 と 17 の
  リガンドでは区間は広くなります。
- cuFEP の値の誤差棒は、その 1 回の計算の標準誤差で、すべての lambda window に共通の時間ブロック
  から求め、解析的な値を下回りません。ブロックが作れないほどサンプルが少ない leg では解析的な値です
  (TYK2 の 16 リガンド中 5、A2A の 17 化合物中 5)。同じ complex leg を繰り返し計算した値どうしは、この標準誤差より大きく違ったことが
  あります ([protocol: REST2](https://github.com/ytworks/cuFEP/blob/main/docs/protocol.md#rest2))。
  誤差棒は、2 回目の計算がどれだけ違いうるかを小さく見せます。
- すべての leg に、文献の 2 つのチェックをします。NOT RELIABLE: 隣り合う 2 つの lambda window の
  overlap が 0.03 未満 (Klimovich ら 2015)。few independent samples: ある lambda window が MBAR に
  渡す相関のない標本が 50 未満 (Mey ら 2020)。結果のあるリガンドはすべて、NOT RELIABLE のものも
  含めて統計に入れています。NOT RELIABLE のものは図で白抜きの丸で描き、2 つ目の表でそれ以外の
  リガンドと分けて比較します。

## TYK2

16 リガンドすべてに結合自由エネルギーがあります。NOT RELIABLE のリガンドはありません:
隣り合う lambda window の overlap の最小値は 0.139〜0.180 です。16 のうち 12 では、ある leg に独立
サンプルが 50 未満の lambda window があります。adaptive lambda scheduling は 32 の leg すべてで基準
0.18 で動き、complex leg に 5〜15、solvent leg に 4〜8 の窓を足しました。

![TYK2: 計算値と実験値](images/tyk2_scatter.png)

![TYK2: リガンドごとの値と、実験値との差](images/tyk2_ligands.png)

全 16 リガンドの統計 (kcal/mol。かっこ内は 95 % ブートストラップ区間):

| 統計量 | cuFEP | FEP+ |
|---|---|---|
| 平均符号誤差 (kcal/mol) | -0.62 [-1.44, 0.29] | 0.00 [-0.23, 0.23] |
| RMSE、生 (kcal/mol) | 1.85 [1.33, 2.34] | 0.47 [0.29, 0.65] |
| MUE、生 (kcal/mol) | 1.52 [1.03, 2.05] | 0.37 [0.24, 0.53] |
| 平均誤差を除いた RMSE (kcal/mol) | 1.74 [1.22, 2.12] | 0.47 [0.28, 0.63] |
| ペアワイズ ddG RMSE (kcal/mol) | 2.55 [1.78, 3.10] | 0.69 [0.41, 0.92] |
| Kendall τ | 0.12 [-0.36, 0.54] | 0.81 [0.61, 0.96] |
| Spearman ρ | 0.21 [-0.40, 0.69] | 0.93 [0.75, 0.99] |
| Pearson r | 0.44 [-0.19, 0.77] | 0.93 [0.85, 0.98] |
| Pearson R² | 0.19 [0.00, 0.59] | 0.86 [0.73, 0.96] |

NOT RELIABLE のリガンドがないので、分けて比べる 2 つ目の群はありません。

すべてのリガンド (kcal/mol。adaptive scheduling の後の complex leg と solvent leg の lambda window
の数。チェックのフラグ):

| リガンド | 実験値 | cuFEP | 標準誤差 | cuFEP − 実験値 | FEP+ | FEP+ − 実験値 | 窓 complex / solvent | 最小 overlap | フラグ |
|---|---|---|---|---|---|---|---|---|---|
| jmc_23 | -11.70 | -9.60 | 0.37 | +2.10 | -11.27 | +0.43 | 25 / 17 | 0.145 |  |
| ejm_46 | -11.31 | -11.26 | 0.48 | +0.05 | -10.82 | +0.49 | 22 / 17 | 0.158 | few independent samples: solvent leg |
| jmc_27 | -11.28 | -9.48 | 0.52 | +1.80 | -11.10 | +0.18 | 25 / 17 | 0.154 | few independent samples: solvent leg |
| jmc_28 | -10.98 | -9.76 | 0.33 | +1.22 | -10.34 | +0.64 | 24 / 17 | 0.140 | few independent samples: complex leg |
| jmc_30 | -10.94 | -14.69 | 0.91 | -3.75 | -11.13 | -0.19 | 30 / 18 | 0.164 | few independent samples: solvent leg |
| ejm_54 | -10.53 | -12.47 | 0.57 | -1.94 | -10.38 | +0.15 | 25 / 18 | 0.155 | few independent samples: complex leg |
| ejm_42 | -9.78 | -11.97 | 0.41 | -2.19 | -9.95 | -0.17 | 21 / 20 | 0.139 | few independent samples: solvent leg |
| ejm_47 | -9.70 | -9.41 | 0.31 | +0.29 | -9.94 | -0.24 | 27 / 18 | 0.168 |  |
| ejm_45 | -9.56 | -10.43 | 0.48 | -0.87 | -9.66 | -0.10 | 22 / 18 | 0.170 | few independent samples: complex leg |
| ejm_31 | -9.54 | -9.84 | 0.33 | -0.30 | -9.91 | -0.37 | 20 / 19 | 0.165 | few independent samples: complex leg |
| ejm_55 | -9.21 | -10.23 | 0.46 | -1.02 | -10.34 | -1.13 | 21 / 19 | 0.156 | few independent samples: solvent leg |
| ejm_48 | -9.00 | -10.72 | 0.36 | -1.72 | -8.86 | +0.14 | 24 / 17 | 0.168 |  |
| ejm_50 | -8.98 | -11.04 | 0.54 | -2.06 | -9.40 | -0.42 | 21 / 21 | 0.180 | few independent samples: complex leg |
| ejm_43 | -8.26 | -8.21 | 0.41 | +0.05 | -8.30 | -0.04 | 23 / 18 | 0.166 | few independent samples: solvent leg |
| ejm_49 | -7.75 | -11.07 | 0.60 | -3.32 | -8.05 | -0.30 | 21 / 17 | 0.139 | few independent samples: complex leg, solvent leg |
| ejm_44 | -7.42 | -5.73 | 0.36 | +1.69 | -6.48 | +0.94 | 21 / 20 | 0.162 |  |

数値から言えること (判定ではありません):

- 平均誤差を除いた RMSE 1.74 kcal/mol は、以前 TYK2 の 5 リガンドを 3 回ずつ計算して測った、
  cuFEP の 1 回の結合自由エネルギーの計算ごとの標準偏差 1.6 kcal/mol (以前の版) に近い値です
  ([protocol: Checks of the analysis](https://github.com/ytworks/cuFEP/blob/main/docs/protocol.md#checks-of-the-analysis))。
- ずれが最も大きいのは jmc_30 (-3.75)、ejm_49 (-3.32)、ejm_42 (-2.19)、jmc_23 (+2.10)、
  ejm_50 (-2.06) です。
- 実験値で最も強く結合する jmc の 3 リガンド (jmc_23、jmc_27、jmc_28) は、cuFEP では実験値より
  1.2〜2.1 kcal/mol 弱く出ており、順位相関を下げています。

## A2A

17 化合物すべてに結合自由エネルギーがあります。NOT RELIABLE の化合物はありません: 隣り合う
lambda window の overlap の最小値は 0.035〜0.182 で、4e の 0.035 が基準 0.03 に最も近い値です。
17 のうち 16 で、complex leg に独立サンプルが 50 未満の lambda window があり、4i、4h、4f、4q、4j では
8 化合物で 4〜13 しかありません (4i 4、4h 6、4f 9、4e 10、4g 10、4o 11、4q 11、4j 13)。adaptive lambda scheduling は 34 の leg すべてで基準 0.18 で動き、complex leg
に 2〜6、solvent leg に 3〜7 の窓を足しました。

![A2A: 計算値と実験値](images/a2a_scatter.png)

![A2A: 化合物ごとの値と、実験値との差](images/a2a_ligands.png)

全 17 化合物の統計 (kcal/mol。かっこ内は 95 % ブートストラップ区間):

| 統計量 | cuFEP | FEP+ |
|---|---|---|
| 平均符号誤差 (kcal/mol) | -4.12 [-5.15, -3.01] | -0.16 [-0.68, 0.36] |
| RMSE、生 (kcal/mol) | 4.70 [3.73, 5.53] | 1.09 [0.68, 1.43] |
| MUE、生 (kcal/mol) | 4.26 [3.30, 5.19] | 0.84 [0.53, 1.18] |
| 平均誤差を除いた RMSE (kcal/mol) | 2.26 [1.47, 2.85] | 1.08 [0.58, 1.40] |
| ペアワイズ ddG RMSE (kcal/mol) | 3.30 [2.14, 4.15] | 1.58 [0.85, 2.05] |
| Kendall τ | 0.38 [-0.04, 0.74] | 0.59 [0.28, 0.86] |
| Spearman ρ | 0.48 [-0.09, 0.85] | 0.73 [0.35, 0.95] |
| Pearson r | 0.39 [-0.19, 0.74] | 0.71 [0.28, 0.90] |
| Pearson R² | 0.15 [0.00, 0.55] | 0.50 [0.08, 0.81] |

NOT RELIABLE の化合物がないので、分けて比べる 2 つ目の群はありません。

すべての化合物 (kcal/mol。adaptive scheduling の後の complex leg と solvent leg の lambda window の
数。チェックのフラグ):

| リガンド | 実験値 | cuFEP | 標準誤差 | cuFEP − 実験値 | FEP+ | FEP+ − 実験値 | 窓 complex / solvent | 最小 overlap | フラグ |
|---|---|---|---|---|---|---|---|---|---|
| 4e | -12.07 | -18.91 | 0.52 | -6.83 | -11.01 | +1.06 | 18 / 17 | 0.035 | few independent samples: complex leg |
| 4n | -11.55 | -18.50 | 0.65 | -6.95 | -13.93 | -2.38 | 20 / 19 | 0.155 | few independent samples: complex leg |
| 4k | -11.54 | -15.54 | 0.34 | -4.00 | -12.00 | -0.46 | 20 / 18 | 0.171 |  |
| 4r | -11.46 | -16.43 | 0.63 | -4.96 | -12.31 | -0.85 | 21 / 18 | 0.166 | few independent samples: complex leg |
| 4c | -11.46 | -15.14 | 0.72 | -3.68 | -12.09 | -0.63 | 18 / 19 | 0.137 | few independent samples: complex leg |
| 4f | -11.45 | -14.52 | 0.38 | -3.08 | -11.72 | -0.27 | 21 / 20 | 0.154 | few independent samples: complex leg |
| 4l | -11.38 | -13.82 | 0.65 | -2.44 | -11.32 | +0.06 | 17 / 17 | 0.140 | few independent samples: complex leg |
| 4o | -11.36 | -10.19 | 0.85 | +1.16 | -9.44 | +1.92 | 18 / 16 | 0.169 | few independent samples: complex leg |
| 4g | -11.06 | -13.57 | 0.69 | -2.51 | -11.83 | -0.77 | 19 / 17 | 0.175 | few independent samples: complex leg |
| 4j | -10.89 | -14.63 | 0.47 | -3.75 | -11.54 | -0.65 | 18 / 17 | 0.121 | few independent samples: complex leg |
| 4h | -10.65 | -18.36 | 1.02 | -7.70 | -11.13 | -0.48 | 17 / 19 | 0.102 | few independent samples: complex leg |
| 4m | -10.55 | -16.30 | 0.30 | -5.75 | -11.71 | -1.16 | 18 / 17 | 0.173 | few independent samples: complex leg |
| 4d | -10.51 | -13.30 | 0.34 | -2.80 | -9.98 | +0.53 | 18 / 18 | 0.182 | few independent samples: complex leg |
| 4i | -10.31 | -15.41 | 0.66 | -5.10 | -10.97 | -0.66 | 18 / 17 | 0.107 | few independent samples: complex leg |
| 4b | -9.96 | -16.68 | 0.62 | -6.72 | -10.06 | -0.10 | 18 / 17 | 0.118 | few independent samples: complex leg |
| 4a | -9.41 | -10.15 | 0.45 | -0.74 | -7.15 | +2.26 | 17 / 18 | 0.115 | few independent samples: complex leg |
| 4q | -8.89 | -13.09 | 0.36 | -4.20 | -8.96 | -0.07 | 18 / 18 | 0.164 | few independent samples: complex leg |

数値から言えること (判定ではありません):

- 17 化合物のうち 16 が実験値より強く結合する側に出ています。17 化合物全体の平均符号誤差は
  -4.12 kcal/mol です。
  Felis の論文も、この標的について同じ向きのオフセット 2.74 を報告しています。
- example のリガンド 4e は、ここでは -18.91 ± 0.52 kcal/mol でした。以前のコミットで同じ設定で
  1 回計算したときは -11.95 ± 0.73 でした。違いは complex leg にあり (今回 44.68、以前 37.61)、
  solvent leg は同じです (16.67、16.52)。
- 二重膜の 10 ns の平衡化の最後の 2 つの四半分の間で、面積は
  -2.3〜+0.1 %、リガンド近くの水の数は -1.7〜+1.6 変化しています (17 の complex leg)。

## 別の ABFE 手法 Felis との比較

Felis (ByteDance Seed) は OpenMM で絶対結合自由エネルギーを計算する手法で、同じ公開セットで評価
されています (Ding ら、arXiv:2603.22274、2026 年 11 月 9 日の版 3)。リガンドごとの値は公開されて
いないため、比較は標的ごとに、その論文の定義による統計量で行います: R² (Pearson)、Spearman ρ、
Kendall τ、オフセット b = ⟨実験値 − 予測値⟩、b だけずらした予測値の RMSE と MAE、実験値の差が
2 kcal/mol を超えるペアのうち順序が正しいものの割合。cuFEP と FEP+ の行はこの定義でここで計算した
ものです。公開されている FEP+ の予測値にこの定義を当てはめると、両標的とも論文の FEP+ の行と
一致します。Felis の行は論文から写したものです。

| | cuFEP (このページ) | Felis (論文) |
|---|---|---|
| リガンドあたりの計算 | 1 回 | 3 回の平均 |
| λ 状態の数、complex / solvent | adaptive: TYK2 20-30 / 17-21、A2A 17-21 / 16-20 | 中性リガンドで 80 / 24 (電荷のあるリガンドは 96 / 32) |
| 状態あたりのサンプリング | 平衡化 1 ns の後 5 ns | 5 ns (またはその最初の 3 ns)、破棄なし |
| 拡張サンプリング | Hamiltonian レプリカ交換、ポケットに REST2 | Hamiltonian レプリカ交換、リガンド近くの水に RelocationMC |
| リガンドの力場 | OpenFF 2.2.1、AM1-BCC | ByteFF25、AM1-BCC (または ABCG2) |
| 推定法 | MBAR | 隣り合う状態間の BAR |
| リガンドの状態 | 化合物ごとに 1 レコード: A2A は FEP+ の結果ファイルが報告する中性のレコード (29 レコード中 17) | ベンチマークのすべてのレコード (別の結合ポーズとプロトン化状態、857 エントリーで 998 配置) を計算し、化合物ごとに対称性と pKa/互変異性の補正で 1 つにまとめる |
| 結晶水 | 入力にあるもの。ここで使った TYK2 と A2A のタンパク質ファイルには水がない | ベンチマークのリポジトリにあるものをそのまま残す |
| A2A の環境 | cuFEP が作る POPC 二重膜に受容体を入れる | 論文に記載がない |

TYK2:

| 手法 | リガンド | R² | Spearman ρ | Kendall τ | オフセット b | 補正後 RMSE | 補正後 MAE | 順序正解率 |
|---|---|---|---|---|---|---|---|---|
| cuFEP、この計算 (5 ns × 1 回) | 16 | 0.19 | 0.21 | 0.12 | 0.62 | 1.74 | 1.49 | 0.68 |
| FEP+、公開予測値からここで計算 | 16 | 0.86 | 0.93 | 0.81 | 0.00 | 0.47 | 0.37 | 1.00 |
| Felis、5 ns × 3 回、ByteFF25 AM1-BCC |  | 0.69 | 0.82 | 0.68 | 0.04 | 0.82 | 0.56 | 0.97 |
| Felis、3 ns × 3 回、ByteFF25 AM1-BCC |  | 0.64 | 0.83 | 0.67 | 0.23 | 0.89 | 0.64 | 0.97 |
| Felis、5 ns × 3 回、ByteFF25 ABCG2 |  | 0.39 | 0.66 | 0.48 | 0.09 | 1.06 | 0.76 | 0.91 |
| FEP+ (論文の表の値) |  | 0.86 | 0.93 | 0.81 | 0.00 | 0.47 | 0.37 | 1.00 |

A2A:

| 手法 | リガンド | R² | Spearman ρ | Kendall τ | オフセット b | 補正後 RMSE | 補正後 MAE | 順序正解率 |
|---|---|---|---|---|---|---|---|---|
| cuFEP、この計算 (5 ns × 1 回) | 17 | 0.15 | 0.48 | 0.38 | 4.12 | 2.26 | 1.79 | 0.94 |
| FEP+、公開予測値からここで計算 | 17 | 0.50 | 0.73 | 0.59 | 0.16 | 1.08 | 0.79 | 1.00 |
| Felis、5 ns × 3 回、ByteFF25 AM1-BCC |  | 0.55 | 0.81 | 0.65 | 2.74 | 1.16 | 0.88 | 1.00 |
| Felis、3 ns × 3 回、ByteFF25 AM1-BCC |  | 0.53 | 0.75 | 0.57 | 2.98 | 1.20 | 0.92 | 1.00 |
| Felis、5 ns × 3 回、ByteFF25 ABCG2 |  | 0.57 | 0.87 | 0.72 | 2.04 | 1.07 | 0.75 | 0.94 |
| FEP+ (論文の表の値) |  | 0.50 | 0.73 | 0.59 | 0.16 | 1.08 | 0.79 | 1.00 |

## この結果から言えないこと

- リガンドごとに 1 回の計算: 同じセットをもう一度計算すると、上に書いたばらつきの範囲で別の数値に
  なります。GPU 上の MD はビット単位では再現しません。
- 標的 2 つ、リガンド 33: 区間は広く、A2A は実験値の範囲が狭いため順位相関の余地が小さくなります。
- FEP+ の値は別の力場とプロトコル、相対計算によるものです。2 つの手法の違いには、力場、
  サンプリング、プロトコルの違いが混ざっています。
- 電荷を持つリガンド、補因子、ほかの標的については何も言えません。このページは実行時間を示しません
  ([docs/LICENSING.md](https://github.com/ytworks/cuFEP/blob/main/docs/LICENSING.md))。

## 結果の再現

上のコミットから作ったイメージ (`docker/Dockerfile`) で、マシンのすべての GPU を使って:

```bash
cufep benchmark run tyk2 --tier full -o runs/benchmark_tyk2_full
cufep benchmark run a2a --tier full -o runs/benchmark_a2a_full
cufep benchmark report tyk2 runs/benchmark_tyk2_full
cufep benchmark report a2a runs/benchmark_a2a_full
```

レポートは実行ディレクトリに `benchmark_report.md`、`benchmark_report.csv`、`benchmark_report.png`、
`benchmark_report_ligands.png` を書きます。記録したものは [benchmarks/results](https://github.com/ytworks/cuFEP/tree/main/benchmarks/results)
にあります。

## 出典

- Ross ら, "The maximal and current accuracy of rigorous protein-ligand binding free energy
  calculations", Commun. Chem. 2023, 6, 222。ベンチマーク:
  https://github.com/schrodinger/public_binding_free_energy_benchmark (MIT ライセンス)。
- 2 つのセットの出典 (ベンチマークが示すもの): doi:10.1021/ja512751q (TYK2)、
  doi:10.1021/acs.jcim.0c00449 (A2A)。
- Klimovich, Shirts, Mobley, "Guidelines for the analysis of free energy calculations",
  J. Comput. Aided Mol. Des. 2015, 29, 397.
- Mey ら, "Best Practices for Alchemical Free Energy Calculations", Living J. Comput. Mol. Sci.
  2020, 2, 18378.
- Ding, Liu, Wang, Xia, Wang, Yan, "Development and large-scale benchmarks of a protein-ligand
  absolute binding free energy toolkit", arXiv:2603.22274 (版 3、2026 年 11 月 9 日)。
  https://github.com/ByteDance-Seed/felis
- Midgley, Bariami, Habgood, Mackey, "Adaptive Lambda Scheduling: A Method for Computational
  Efficiency in Free Energy Perturbation Simulations", J. Chem. Inf. Model. 2025, 65, 512.

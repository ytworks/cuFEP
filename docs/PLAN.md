# cuFEP 実装計画 (改訂 7 の公開版)

オーナーが 2026-11-19 に承認した再スタート計画の要約。仕様の本文は
[protocol.md](protocol.md) と [configuration.md](configuration.md) にある。
cuPotential の速度に関わる実測値と所要時間の目標値は、NVIDIA のライセンス
(性能の開示には書面の許可が必要) に従い、このリポジトリには載せない。

## 方針

1. **世間の FEP の仕様と用語に合わせる。** 既定値と語は Schrödinger FEP+、Cresset Flare、
   OpenFE のどれに従ったかを protocol.md の比較表に書く。1 つの計算は既定で 1 回だけ走らせる
   (FEP+ と Flare と同じ)。`sampling.repeats` で独立な repeat を増やせる (OpenFE と同じ意味)。
2. **すべての設定は YAML で変えられ、既定値は根拠のある推奨値である。** lambda window は
   leg と stage ごとに、数または値の列で指定できる。`auto` は入力から計算しないと決まらない
   項目だけに使う。
3. **lambda window の既定は adaptive lambda scheduling** (Midgley ら、J. Chem. Inf. Model.
   2025, 65, 512。Cresset Flare の方式)。window が差し込まれる条件は protocol.md に論文の
   記述どおりに書いた。論文に無い点は、無いと明記して規則を決めた。
4. **速度は要件である。** replica を全 GPU に配り、NVIDIA MPS で GPU 1 個に複数の worker を
   同時に載せる (SOMD2 の oversubscription factor、Desmond の `mps_factor` と同じ方式)。
   GPU の配分は job の残りの仕事量に比例させ、空いた分は動いている job に渡す。
5. **膜は設定で付ける** (`membrane.enabled`)。膜タンパク質かどうかの判定はしない。
6. **自動で決めるのは名指しされたものだけ**: Boresch restraints、lambda window (adaptive)、
   GPU の割当。失敗からの回復 (stock OpenMM への退避、2 fs への退避、worker の再起動) は
   必ずログに残す。production の自動延長はしない。

## 手順と状態

| # | 作業 | 状態 |
|---|---|---|
| 0 | 旧方式の実行を止める。OpenFE の参照計算は続ける | 済 |
| 1 | 膜判定を削除し、`membrane.enabled` にする | 済 |
| 2a | 状態エネルギーの正しさ: `setPositions` の直後に soft-core の力グループだけを評価すると neighbor list が古いまま使われる問題を特定して修正 (M0_REPORT.md) | 済 |
| 2b | TYK2 ejm_31 の complex leg が repeat の間で 4.6 kcal/mol 開く原因の調査 | 原因を特定 (下記。complex leg の側鎖の回転異性体と、solvent leg のリガンドの二面角)。対策はオーナーと決める |
| 3 | 速度の実測: MPS の oversubscription、iteration の内訳、GPU 数に対するスケーリング、pre-equilibration | 済 (GPU 8 個と膜系は後の手順で測る) |
| 4 | 設定スキーマを作り替える。fixed の推奨値を設定の `windows` で与える | 済 |
| 5 | GPU の割当を作り替える。MPS、仕事量に比例した配分、worker の増減、`timing.json` | 済 |
| 6 | adaptive lambda scheduling を実装する | 済 |
| 7 | 文書を仕様と用語に合わせる。`docs/configuration.md` を新設 | 済 |
| 8 | TYK2 example を既定設定で GPU 8 個と 1 個で走らせ、所要時間を測る | GPU 1 個と 5 個で実測済み。GPU 8 個は OpenFE の参照計算 (GPU 3 個を使用中) が終わってから測る |
| 9 | プロトコルの検証: TYK2 と T4 リゾチームを 3 repeats で、adaptive と fixed の両方。restraint の強さ、pre-equilibration の長さの比較 | TYK2 は済。adaptive の開始 window 数を実測で確定 (protocol.md)。restraint の強さ (固定と自動) と pre-equilibration (5 ns と 1 ns) は TYK2 で差が出ず、既定は変えない (configuration.md)。T4 リゾチーム/トルエンは 9 本中 6 本が -4.2〜-5.2 (実験 -5.52)、3 本は対称なリガンドの反転で無効 (下記) |
| 10 | A2A example (`membrane: {enabled: true}`) | 未 |
| 11 | OpenFE との照合 (TYK2 ejm_31、jmc_23、T4 リゾチーム/トルエン) | 参照計算を実行中 |
| 12 | ベンチマーク (smoke、subset、full)。1 リガンド 1 回 | 未 |
| 13 | 公式イメージの作り直しと確認 | 未 |

## T4 リゾチーム L99A / トルエン (手順 9)

実験値は -5.52 kcal/mol。3 repeats、production 5 ns。

| 実行 | dG_bind (kcal/mol)。括弧内は restrain stage の最初の 2 つの window の overlap |
|---|---|
| adaptive (既定)。restraint の基準値が軌跡の平均だった版 | -5.2 (0.03)、-4.2 (0.02)、-5.0 (0.10) |
| adaptive (既定)。restraint の基準値を最終フレームにした版 | -4.5 (0.13)、-4.7 (0.10)、無効 (0.00) |
| fixed (30 / 14 windows)。基準値が軌跡の平均だった版 | -4.3 (0.15)、無効 (0.00)、無効 (0.00) |

無効と書いた 3 本は、その overlap が 0 で、出てきた値 (-44.8、-16.5、-20.2) に意味が無い。
overlap が 0.03 を下回る実行では、cuFEP は結果の行に `NOT RELIABLE` と警告を出す。

この系で見つけて直したもの:

- Boresch restraint の基準値 (距離と角度) を軌跡の平均から取っていた。空洞の中で位置を変える
  リガンドでは、平均は実際には取らない構造になる。OpenFE と同じく最終フレームの値にした。
- どの window とも重ならない window があると MBAR が収束判定に達せず、解析が 14 分と 53 分
  かかった。反復回数の上限が実際に効くようにし、10〜15 秒になった。

残っている制約:

- トルエンは空洞の中で向きを変える (対称な反転を含む)。restraint を掛けていない window
  (restrain stage の lambda 0) がいったん別の向きに移ると、restraint のエネルギーが 100〜660 kT
  になり、restraint を掛けた window と重ならなくなる。complex leg 9 本のうち 7 本で実行の
  12 分の 1 以上の時間にわたって起き、うち 3 本は戻らずに無効になった。対称な小さいリガンドに Boresch restraints を
  使うときの既知の問題 (Mobley, Chodera, Dill, J. Chem. Phys. 2006, 125, 084902) で、OpenFE の
  方式にも同じ制約がある。TYK2 の ejm_31 では起きていない (全ての実行で restrain stage の
  overlap は 0.2 以上)。
- 対策の候補 (未決定。オーナーが決める): リガンドの対称な原子の入れ替えを考慮した restraint と
  対称性の補正を実装する。または、検証の系を対称でないリガンドに替える。

## 2b の調査結果

TYK2 ejm_31 を旧スケジュール (28〜31 windows、production 5 ns) で 3 repeats 走らせた結果:

| repeat | complex leg (kcal/mol) | solvent leg | dG_bind |
|---|---|---|---|
| 0 | 43.6 | 20.3 | -14.0 |
| 1 | 38.9 | 20.6 | -9.0 |
| 2 | 42.6 | 20.2 | -13.1 |

実験値は -9.54、FEP+ は -9.91 kcal/mol。solvent leg は 0.4 kcal/mol 以内で一致し、開きは
すべて complex leg にある。リガンドの姿勢、ヒンジとの水素結合、リガンドの二面角は 3 回とも
同じだった。違うのはポケットの ILE960 の chi1 (VAL981、GLU979 の側鎖と連動) である。

- 結晶構造と repeat 0、2 の開始構造は trans。repeat 1 は pre-equilibration の最後に g- へ
  移り、全 window が g- から始まった。
- リガンドの無い側の window は、trans から始めても数 ns かけて g- に移っていく。
- 1 ns ごとの解析で、trans の 2 回は discharge が約 22.5、vanish が約 14.2、g- の 1 回は
  約 20.3 と約 12.3 kcal/mol で、差は時間が経っても縮まらない。
- complex leg では replica が両端まで行き来できていない (5 ns で両端に達した replica は
  2〜6 本、往復は 0)。solvent leg ではほぼ全ての replica が両端に達している。

つまり、リガンドの有無で安定な向きが変わるポケットの側鎖が数 ns では入れ替わらず、
開始構造の向きが結果に残る。

### 手順 9 で分かったこと (2026-11-19)

lambda window を作り替えた後の 3 repeats (TYK2 ejm_31、production 5 ns):

| lambda window | complex leg | solvent leg | dG_bind (kcal/mol) | 平均 ± SD |
|---|---|---|---|---|
| 旧方式 (28〜31 / 31〜34 windows) | 43.6、38.9、42.6 | 20.3、20.6、20.2 | -14.0、-9.0、-13.1 | -12.1 ± 2.7 |
| fixed (OpenFE と同じ 30 / 14 windows) | 40.5、41.8、38.9 | 20.8、20.0、20.7 | -10.5、-12.5、-9.0 | -10.7 ± 1.8 |
| adaptive (既定。19 / 21 windows) | 41.1、39.0、41.3 | 19.9、21.4、20.6 | -11.9、-8.4、-11.5 | -10.6 ± 1.9 |

window の数と overlap を変えても repeat の間の開きは変わらない。adaptive の既定では、両 leg の
全ての隣り合う window で overlap 0.13 以上、交換 5% 以上になっている。以前に候補に挙げた
「window を減らして replica の往復を速くする」は、この結果から対策にならない。

solvent leg にも遅い自由度がある。リガンドのアセトアミド基がピリジン環に対して反転する
二面角で、水中では数 ns に 1 回しか入れ替わらない。solvent leg の値は、結合時の向きのまま
だった実行で約 20.0、反転した実行で約 20.9 kcal/mol である (13 回の実行で相関 0.8。window の
数 13〜34 には依らない)。OpenFE の参照計算の solvent leg も 3 repeats の間で 1.4 kcal/mol
開いている。complex leg ではこの二面角は結合時の向きに固定されている。

FEP+ は結合部位とリガンドの REST2 と水の GCMC でこの種の問題に対処している。OpenFE は
3 repeats の平均と標準偏差を報告する。cuFEP の対策の候補は次のとおりで、どれを採るかは
未決定 (オーナーが決める)。

1. repeat を増やして平均する (`sampling.repeats`。OpenFE の既定は 3)。
2. production を長くする (OpenFE は 10 ns)。
3. 結合部位の側鎖とリガンドに REST2 を入れる (FEP+ の方式)。
4. リガンドの無い側の window の初期構造を、リガンドを消した状態の MD から取る。

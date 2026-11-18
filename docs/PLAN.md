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
| 2b | TYK2 ejm_31 の complex leg が repeat の間で 4.6 kcal/mol 開く原因の調査 | 原因を特定 (下記)。対策はオーナーと決める |
| 3 | 速度の実測: MPS の oversubscription、iteration の内訳、GPU 数に対するスケーリング、pre-equilibration | 済 (GPU 8 個と膜系は後の手順で測る) |
| 4 | 設定スキーマを作り替える。fixed の推奨値を設定の `windows` で与える | 済 |
| 5 | GPU の割当を作り替える。MPS、仕事量に比例した配分、worker の増減、`timing.json` | 済 (検証中) |
| 6 | adaptive lambda scheduling を実装する | 済 |
| 7 | 文書を仕様と用語に合わせる。`docs/configuration.md` を新設 | 済 |
| 8 | TYK2 example を既定設定で GPU 8 個と 1 個で走らせ、所要時間を測る | 未 |
| 9 | プロトコルの検証: TYK2 と T4 リゾチームを 3 repeats で、adaptive と fixed の両方。restraint の強さ、pre-equilibration の長さの比較 | 未 |
| 10 | A2A example (`membrane: {enabled: true}`) | 未 |
| 11 | OpenFE との照合 (TYK2 ejm_31、jmc_23、T4 リゾチーム/トルエン) | 参照計算を実行中 |
| 12 | ベンチマーク (smoke、subset、full)。1 リガンド 1 回 | 未 |
| 13 | 公式イメージの作り直しと確認 | 未 |

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
開始構造の向きが結果に残る。FEP+ は結合部位の REST2 と水の GCMC でこの種の問題に対処
している。cuFEP の対策の候補は次のとおりで、どれを採るかは未決定。

1. window を減らして replica の往復を速くする (adaptive lambda scheduling で 15 windows 前後に
   なる)。手順 9 で repeat の間の開きを測り直す。
2. production を長くする。
3. 結合部位の側鎖とリガンドに REST2 を入れる (FEP+ の方式)。
4. リガンドの無い側の window の初期構造を、リガンドを消した状態の MD から取る。

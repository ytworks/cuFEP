# cuFEP 実装計画 (改訂 2): cuPotential を既定バックエンドとする OpenMM ABFE エンジン

## Context

リポジトリ `/home/y-takagi/cuFEP` (origin: `ytworks/cuFEP`) に、OpenMM ベースの絶対結合自由エネルギー (ABFE) エンジンを新規実装する。初版計画は承認済みで、M0 に着手した直後にオーナーから追加要件が出たため再計画する。

**当初の要件**

1. cuPotential を既定として使う OpenMM ベースの ABFE エンジン
2. 公式の動作 Docker
3. 膜タンパク質は脂質二重層を入れて実行できる
4. アダプティブ λ ウィンドウが既定で動く
5. Example を膜系とキナーゼ系で 1 つずつ。ベンチマークも各 1 つ、実験値と FEP+ の両方と比較できるもの
6. 実行設定は YAML

**追加要件 (今回)**

7. 入力はタンパク質の PDB とリガンドの SDF だけで動く。リガンド SDF は対象ポケットの位置にある前提
8. PR を作り、こまめに push する
9. 拘束などの設定を自動で最適に選ぶ機能を持つ
10. OpenFE も参考にして、精度が出る方式にする
11. 1 GPU でもマルチ GPU でも既定で動く

**前提の訂正 (初版から継続)**: cuPotential は力場ではなく、NVIDIA の GPU カーネルライブラリ (OpenMM プラグイン `nv_openmm_plugin`)。本計画では「既定の計算バックエンド」として扱い、力場パラメータは AMBER 系を既定にする。

## 初版からの変更点と、その根拠になった新事実

| 変更 | 新事実 |
|---|---|
| 膜系の入力を「配向済み構造」から「膜の自動判定と自動配向」へ | 追加要件 7 |
| マルチ GPU を後段 (M3) から中核 (M2) へ。既定で全 GPU を自動利用 | 追加要件 11 |
| 「自動設定」層を新設。`auto` を具体値に解決して記録 | 追加要件 9 |
| 拘束の力定数、平衡化、サンプリング量、反復数の既定を OpenFE に合わせる。OpenFE との数値照合を合格条件に追加 | 追加要件 10。OpenFE の既定値を文書で確認 |
| リガンド電荷の既定を NAGL から AM1-BCC (AmberTools `sqm`) へ。NAGL は高速オプション | ビルドしたイメージに AmberTools が既に入っていた。OpenFE の既定も `sqm` |
| プラグインの入手元を conda-forge から NVIDIA のリリース同梱 conda チャネルへ (SHA256 固定) | conda-forge に `nv-openmm-plugin` が存在しなかった |
| 膜構築後に「タンパク質内部の水和」工程を追加 | OpenMM `addMembrane` の実装を確認。膜パッチの z 範囲内には水を足さないため、GPCR のポケットが乾いた状態で始まる |
| コミット、push、PR 作成を行う | 追加要件 8 |

変更していない判断: バックエンド 2 本立て、自前 factory、分離構成、MBAR、自前レプリカ交換、adaptive λ、Boresch 拘束、端点 LRC 再重み付け、TYK2 と A2A のベンチ選定、荷電リガンドは v0.1 対象外。

## 調査で確定した事実

### cuPotential v0.3.5 (early access)

| 項目 | 内容 |
|---|---|
| 導入 | PyPI wheel または NVIDIA リリースの conda チャネル。Python 3.12、`openmm>=8.4,<8.5`、CUDA 13.4.3 |
| ドライバ | R615 以上。プロ/データセンター GPU は R595 等でも `cuda-compat-13-4` で可 |
| 置換部品 | `nv.NonbondedForce` (PME/LJPME)、`nv.LangevinMiddleIntegrator`、`nv.MonteCarloBarostat` (**等方のみ**)。他の力は stock カーネルが同じ System 内で計算 |
| パラメータ | global parameter と particle/exception offset を Context 上で更新可 |
| 非対応 | 融合 custom 式での interaction group、**dU/dλ**、NoCutoff。Ewald 許容誤差 5e-4 固定。箱の各辺は cutoff の 3 倍以上 |
| 既知の問題 | パラメータ更新が O(N) のホスト処理。custom 式の tail は新しい λ 値ごとに CPU で約 3 秒。未最小化構造や原子が重なる終状態付近で `CUDA_ERROR_LAUNCH_FAILED` の報告 |
| 第三者計測 (レプリカ交換形式) | stock 比で約 1.3〜1.7 倍。分離構成が融合構成より速い |
| ライセンス | ランタイム libcupo は NVIDIA SLA。**性能ベンチ結果の開示は書面許可が必要 (2.12 条)** |

### OpenFE の ABFE プロトコル (精度の参照。文書で確認した既定値)

| 項目 | OpenFE 既定 |
|---|---|
| 経路 | 拘束 ON、静電を消去、LJ を脱共役 (Beutler softcore α=0.5)、拘束解放は解析式 |
| Boresch 拘束 | K_r 10 kcal/mol/Å²、角度/二面角 80 kcal/mol/rad²。ホスト原子は主鎖、DSSP フィルタ、RMSF 0.1 nm 以下、距離 0.5〜1.5 nm、`bonded` 戦略 |
| 事前平衡化 | 複合体: NVT 0.25 ns、NPT 0.5 ns、MD 5 ns。この軌跡で拘束を探索 |
| サンプリング | レプリカ交換、2.5 ps/反復、複合体 30 状態、溶媒 14 状態、各状態 平衡化 1 ns + 本番 10 ns、3 反復 |
| 系 | ff14SB、TIP3P、openff-2.2.1、AM1-BCC (AmberTools)、PME 0.9 nm、4 fs、HMR 3 amu、十二面体の箱、0.15 M NaCl |
| 解析 | MBAR、overlap、順/逆収束、リガンド RMSD と重心ドリフト |

### この計算機と M0 の進捗

- RTX PRO 6000 Blackwell Server × 8 (各 97 GB)、ドライバ 595.99.02、Docker + NVIDIA Container Toolkit、128 コア、空き 209 GB。
- **同じユーザーの別プロジェクトのコンテナ (`fmeta-*`、一時停止中) が各 GPU を約 6.5 GB 使用し、MPS デーモンも稼働中。** compute mode は Default。オーナーから「GPU は全部解放して使ってよい」と指示があった (2026-11-18)。全 8 GPU を使う。既存コンテナは一時停止中で計算を使っていないので、妨げになった場合にだけ停止する。
- イメージ `cufep:dev` はビルド済み (OpenMM 8.4、プラグイン 0.3.5、openff-toolkit 0.19、pymbar 4.2、AmberTools、`memembed` を含む)。GPU 上での動作確認は未実施。
- ベンチ入力は取得済み (未コミット)。TYK2: 末端キャップが `UNK` 残基。A2A: 残基 208→219 に鎖の切れ目、Na+ 1 個、結晶水なし、全 17 化合物に中性状態あり。

## 設計判断

| 項目 | 採用 |
|---|---|
| バックエンド | `cupotential` (既定) と `openmm` (stock)。同じ alchemical 仕様を両方で実行。失敗時は stock に自動退避 |
| 力場 (既定) | ff14SB + TIP3P + Lipid21 + OpenFF Sage 2.2.1、AM1-BCC (`sqm`) |
| 力の構成 | 分離構成: nv 標準力 (電荷は offset で λ 制御) + stock の softcore LJ (リガンド×環境) |
| 推定 | MBAR (pymbar 4)。3 反復のばらつきを誤差とする |
| サンプラー | 自前の Hamiltonian レプリカ交換。GPU ごとにワーカープロセス 1 つ |
| λ | adaptive が既定 |
| 拘束 | Boresch、OpenFE 方式の自動探索 |
| 長距離分散 | 結合状態フレームを長 cutoff へ再重み付けする端点補正 |
| 膜 | 自動判定 + 自動配向 + `Modeller.addMembrane` (POPC) + 内部水和 |
| 荷電リガンド | v0.1 は非対応 (明示エラー) |

## 入力契約 (要件 7)

必須入力はタンパク質 PDB とリガンド SDF のみ。最小の YAML は 2 行。

```yaml
system:
  protein: protein.pdb
  ligand: ligand.sdf      # 複数分子を含む SDF は 1 分子ずつ ABFE を実行
```

- `cufep init protein.pdb ligand.sdf` が全項目 `auto` の YAML を生成する。
- **タンパク質**: 水素があれば保持、無ければ pH 7.4 で付加。欠損重原子は PDBFixer で補完。鎖の切れ目は鎖を分割してキャップ (欠損ループは作らない)。ジスルフィド、水、イオンは保持。リガンドと衝突する水は削除。
- **リガンド**: 座標はそのまま使う (ドッキングや再配置はしない)。ポケット内にあるかを検査する。ベンチ入力の実測では、結合リガンドは重原子の全てがタンパク質から 4.5 Å 以内にある。
- **黙って進めず停止する場合**: 未知の HET 残基や修飾残基 (補因子は任意キー `system.cofactors` に SDF で渡す)。リガンドの水素欠落、形式電荷が 0 でない、ポケット外、タンパク質との重度の衝突、共有結合距離の原子。リガンドから約 3 Å 以内の金属。
- **進めるが警告する場合**: リガンド近傍 (約 8 Å) の鎖の切れ目、補完した側鎖、代替配座、水素を付加した滴定残基。割り当てた状態をレポートに明記する。
- 既定値は全て YAML で上書きできる。

## 自動設定 (要件 9)

`auto` の値を具体値に解決し、`resolved_config.yaml` と判断理由のレポートを出力先に保存する。`cufep plan config.yaml` は GPU を使わない部分だけを解決して表示する。「拘束」は Boresch 拘束 (restraint) と結合拘束 (constraint) の両方を対象にする。

| 対象 | 自動選択の内容 | 安全策 |
|---|---|---|
| Boresch 拘束 | 事前平衡化 MD から OpenFE 方式で探索。リガンド側は中心に近い芳香環の結合 3 原子。タンパク質側は主鎖、DSSP、RMSF 0.1 nm 以下、距離 0.5〜1.5 nm。共線性、平坦な角度、角度と二面角の分散で候補を除外し、円周分散の和が最小の組を採用。力定数は OpenFE 既定 | リガンド姿勢が MD 中に不安定なら拘束を確定せず停止 |
| 結合拘束と時間刻み | 4 fs + HBonds + HMR 3 amu。結合状態の事前平衡化で不安定を検出したら 2 fs に自動退避 | 終状態付近の不安定は時間刻みでは直さず、原因を報告 |
| λ ウィンドウ | adaptive (下記) | `schedule.yaml` に保存 |
| 平衡化の打ち切り | 自動検出 (pymbar) | — |
| 本番の長さ | 事前に決めた長さ (5 ns、10 ns) でだけ判定。基準は状態ごとの有効サンプル数、レプリカの往復回数、最小 overlap | 推定値やその誤差を終了の引き金にしない (偏りを避ける)。順/逆の不一致は警告。同じ leg の全反復は同じ長さ |
| 膜 | 膜タンパク質かの判定、配向、箱 | 自前の判定と `memembed` が一致したときだけ自動で進む。不一致や低確信度は停止して明示指定を要求 |
| 箱 | 形 (可溶性は十二面体、膜は直方体)、余白、プラグインの「cutoff の 3 倍」規則 | — |
| GPU 割当 | 空いている可視 GPU を全て使用 (下記) | `resources.gpus` で制限可 |

## ABFE プロトコル (要件 10: OpenFE 準拠 + 追加の精度対策)

- **経路**: complex leg は (A) Boresch 拘束 ON、(B) 静電を消去、(C) softcore で LJ を脱共役。solvent leg は (B)(C)。拘束解放の標準状態補正は数値積分で求め、解析式と照合する。
- **事前平衡化**: OpenFE と同じ長さの NVT、NPT、MD。最小化と事前平衡化は stock で実施。
- **OpenFE に合わせる既定**: 力定数、4 fs/HMR/HBonds、PME 0.9 nm、298.15 K、1 bar、0.15 M、各状態 平衡化 1 ns、本番 5 ns または 10 ns、3 反復。反復は事前平衡化と拘束の選択からやり直す完全に独立な実行とし、反復間の標準偏差を誤差として報告する。
- **OpenFE に加える精度対策**: overlap を均一にする adaptive λ。長距離分散の端点補正。リガンド RMSD、重心ドリフト、ポケット内の水分子数の監視 (解離、姿勢変化、水和の変化を警告)。対称なリガンドでは対称等価な原子を拘束に使わず、対称数を報告。膜系の内部水和。
- リガンド分子内の非結合相互作用は exception として全 λ で保持。真空 leg は不要。

## アダプティブ λ (既定)

1. **Trailblaze**: 結合状態から順に短い MD を走らせ、エネルギー差の標準偏差が目標 (既定 1.0 kT) になる次の λ を探索。各ウィンドウの初期構造も得る。
2. **Refine**: 本番の最初のブロックで隣接 overlap が閾値 (既定 0.03) 未満の区間に中点を挿入。
3. `schedule.yaml` に保存。最終推定は確定後のサンプルのみ。`--reschedule` 無しでは再利用。ベンチは凍結スケジュールをコミット。

## マルチ GPU (要件 11)

- 既定は `resources.gpus: auto` (可視 GPU 全て)。1 GPU でも同じコードが動く。
- **2 段の割当**: ジョブ (リガンド × leg × 反復) の数が GPU 数以上なら 1 ジョブ 1 GPU で並べる。GPU が余るときは 1 ジョブのレプリカを複数 GPU に分散する。
- **構造**: GPU ごとにワーカープロセス (spawn) 1 つ。各ワーカーが Context を 1 つ持ち、担当レプリカを順番に進める。レプリカの座標、速度、箱はコントローラが所有する共有メモリに置き、反復ごとに二重バッファで更新する。交換は熱力学状態の番号を入れ替える方式で、パイプを渡るのはエネルギー (スカラー) のみ。
- **反復はトランザクション**: 全ワーカーの報告が揃ったときだけ確定する (バッファ切替、エネルギー追記、チェックポイントの原子的書込み)。再開時は HDF5 を確定済みの反復まで切り詰める。
- **障害時**: ワーカーは生存信号とタイムアウトで監視する (CUDA の失敗は停止せず固まることがある)。落ちたワーカーの担当分だけを反復開始時のバッファからやり直す。stock への退避は、両バックエンドの λ 間エネルギー差が一致すると実測できた構成に限り、サンプルごとに使用バックエンドを記録する。同じ構造から再びクラッシュする場合は物理的な問題とみなし、停止して報告する。
- **再現性**: ホスト側の乱数は (種、反復、レプリカまたはペア) で決め、ワーカー番号に依存させない。目標はビット一致ではなく統計的一致とし、1 GPU と複数 GPU の ΔG が反復ばらつきの範囲で一致することをテストする。
- **共用環境への配慮**: 割当は GPU の実負荷を見て決める。MPS には依存せず、パイプディレクトリを専用の空ディレクトリに向ける。ワーカーごとの CPU スレッド数を制限する。

## 膜タンパク質 (要件 3、7)

- **判定と配向**: 溶媒露出した残基の疎水性に基づく疎水スラブ探索 (法線、中心、厚み) を自前実装。スラブを貫通するヘリックス 2 本以上またはバレルを要件にする。イメージ内の `memembed` と傾きと深さが一致したときだけ自動で進み、法線を z、中心を z=0 に回転する。POPC は対称なので上下の向きは問わない。融合タンパク質 (bRIL、T4L など) は警告する。
- **構築**: リガンドのテンプレートを登録してから `Modeller.addMembrane` (POPC)。続けてタンパク質内部の空洞に水を配置 (結晶水があれば優先)。
- **構築後の検査**: 上下の脂質数の釣り合い、疎水層に孤立した水、ポケットや環を貫く脂質尾部、横方向の脂質の余白。
- **平衡化**: stock OpenMM。段階的拘束解除の後、`MonteCarloMembraneBarostat` (張力 0) で無拘束 10 ns。面積/脂質とポケット内の水分子数が収束していなければ 30 ns まで延長。
- **本番の圧力制御** (M0 で確定。bilayer に等方バロスタットは使わない)
  1. stock `MonteCarloMembraneBarostat` + nv 力 + nv integrator
  2. 同上で stock integrator
  3. 反復間の Python 実装の半等方 MC バロスタット
  4. 平衡化後の平均箱で NVT
  5. 膜系のみ stock バックエンド

## アーキテクチャ

```
cuFEP/
├── pyproject.toml, docker/{Dockerfile, environment.yml, entrypoint.sh}   # 作成済み
├── docker/Dockerfile.openfe-ref      # OpenFE 照合用 (検証専用)
├── src/cufep/
│   ├── cli.py                    # init / plan / run / analyze / validate / check-env / benchmark
│   ├── config/{schema,auto}.py   # pydantic v2。auto.py が自動設定を解決
│   ├── prep/{protein,ligand,forcefield,solvate,membrane,orient,hydrate}.py
│   ├── alchemy/{factory,restraints,schedule,lrc}.py
│   ├── backends/{base,cupotential,openmm_stock}.py
│   ├── sampling/{equilibrate,hrex,worker,scheduler,storage}.py
│   ├── analysis/{mbar,diagnostics,structure,report}.py
│   └── benchmarks/{registry.yaml, curate.py, runner.py, stats.py}
├── spikes/                       # M0 の検証スクリプト
├── examples/{kinase_tyk2, membrane_a2a}/
├── benchmarks/{tyk2, a2a}/       # 入力取得済み (MIT 表示付き)
├── docs/{PLAN.md, LICENSING.md, protocol.md}
└── tests/{unit, gpu, physical}
```

## YAML (全項目。既定は右のとおりで、通常は `system` だけ書けばよい)

```yaml
version: 1
system: {protein: protein.pdb, ligand: ligand.sdf, cofactors: []}
membrane: {enabled: auto, lipid: POPC, padding_nm: 1.5}
forcefield: {protein: amber14/protein.ff14SB.xml, water: amber14/tip3p.xml,
             lipid: amber19/lipid21.xml, ligand: openff-2.2.1, ligand_charges: am1bcc}
backend: {name: cupotential, fallback: openmm, precision: mixed}
thermodynamics: {temperature_K: 298.15, pressure_bar: 1.0, ionic_strength_M: 0.15}
dynamics: {timestep_fs: auto, constraints: auto, hmr_mass_amu: 3.0}
restraints: {type: boresch, selection: auto, force_constants: openfe}
lambda_schedule: {mode: adaptive, target_std_kT: 1.0, min_overlap: 0.03,
                  max_windows: {complex: 48, solvent: 32}}
sampling: {pre_equilibration_ns: 5, equilibration_ns: 1, production_ns: [5, 10],
           min_effective_samples: 50, min_round_trips: 2, repeats: 3}
analysis: {estimator: mbar, lrc: reweight}
resources: {gpus: auto}
output: {dir: runs/example}
```

## Docker

- `docker/Dockerfile` (作成済み、ビルド成功): `nvcr.io/nvidia/cuda:13.4.3-base-ubuntu24.04` + micromamba の固定環境。プラグインは NVIDIA リリースの conda チャネルを SHA256 検証して導入。ENTRYPOINT は `cufep`。`WITH_CUPOTENTIAL=0` で stock 版。
- 動作確認: `docker run --gpus all cufep check-env`。
- 配布: Dockerfile からのビルドを公式手順とする。ビルド済みイメージの GHCR 公開は NVIDIA SLA 確認後のオプション。stock 版は制約なく公開可。

## Example とベンチマーク

| | キナーゼ系 | 膜系 |
|---|---|---|
| 標的 | TYK2 (PDB 4GIH) | アデノシン A2A 受容体 (PDB 5OLZ) |
| 出典 | Schrödinger 公開ベンチ `jacs_set/tyk2` | 同 `gpcrs/a2a_hip278` |
| リガンド | 16 (全て中性) | 17 化合物 (各 1 つの中性状態を選定) |
| 実験 ΔG 範囲 | −11.7 〜 −7.4 kcal/mol | −12.1 〜 −8.9 kcal/mol |
| FEP+ 参照値 | `tyk2_out.csv` | `a2a_hip278_sbpkacorr_out.csv` |

- どちらも「PDB + SDF のみ」の入力契約でそのまま動かす (膜の判定と配向も自動)。データ固有の整形 (TYK2 のキャップ命名、A2A の鎖分割と状態選定) は `benchmarks/curate.py` に閉じ、選定根拠を CSV に記録する。
- **比較**: cuFEP のみ、ゼロ点合わせ無しの平均符号付き誤差、RMSE、MUE。cuFEP と FEP+ の両方、平均誤差を引いた後の RMSE、ペアワイズ ΔΔG RMSE、Kendall τ、Spearman。全て bootstrap 95% 信頼区間付き。優劣は信頼区間で記述する。
- **Tier**: smoke (1 リガンド、極短時間) / subset (実験値の順位で等間隔の 5 リガンド、実行前に固定) / full (全リガンド、3 反復)。full は最大で数十 µs 規模。所要時間は M0 の実測から算出して提示する。
- 公開レポートは精度のみ。cuPotential の速度比較は含めない。

## 開発の進め方 (要件 8)

- ブランチ `feat/abfe-engine` を `main` から作成。現在の未コミット分 (Docker、骨格、ベンチ入力、`docs/PLAN.md`) を最初のコミットとして push し、**ドラフト PR を 1 本**作成する。PR 本文にマイルストーンのチェックリストを置く。
- 手段: この計算機の `gh` CLI (アカウント `ytworks` でログイン済み) で PR を作成する。git の作者情報は未設定なので、既存コミットと同じ作者 (`ytworks`) をコミットごとに指定し、グローバル設定は変更しない。PR テンプレートは無い。
- push の単位: テストが通る作業単位ごと。少なくとも各マイルストーンの小項目ごと。検証結果 (スパイクの判定、ベンチのレポート) も要約をコミットする。`runs/` と `spikes/out/` は対象外。
- CI (GitHub Actions、CPU): lint と単体テスト (stock バックエンド)。GPU テストはこの計算機で実行し、結果を PR に記録する。

## マイルストーン

| # | 内容 | 完了条件 |
|---|---|---|
| M0 | ブランチとドラフト PR。リスク検証スパイク | 下記の実証項目が全て判定済み |
| M1 | YAML スキーマ、自動設定の静的部分、CLI (`init`/`plan`)、タンパク質とリガンドの前処理、可溶性系の構築、stock バックエンド、factory、拘束、MBAR | CPU 単体テスト合格。TYK2 の PDB + SDF から系が組める |
| M2 | cuPotential バックエンド、ワーカープロセス型レプリカ交換 (1 GPU と複数 GPU)、再開、障害時の退避 | 両バックエンドのエネルギー一致。1 GPU と複数 GPU の結果一致。水和自由エネルギーの一致 |
| M3 | adaptive λ、拘束の自動探索、時間刻みの自動退避、収束による延長、構造 QC | T4 リゾチーム/トルエンの ABFE が完走し文献値と整合 |
| M4 | OpenFE との数値照合、公式 Docker、TYK2 example、ドキュメント | 下記の照合基準を満たす。コンテナ内で TYK2 example が完走 |
| M5 | 膜の判定と配向、構築、内部水和、平衡化、圧力制御、A2A example | 配向が OPM と一致。A2A example が完走 |
| M6 | ベンチ (整形、ランナー、レポート) | smoke と subset を実行しレポート生成。full は見積りを提示して実行 |

**M0 で実証すること**

1. コンテナ内で CUDA プラットフォームとプラグインが R595 ドライバ上で動く
2. TYK2 複合体の alchemical System が全 λ 点で stock と一致 (λ 間エネルギー差 0.02 kT 以内)
3. 完全脱共役状態で 1 ns 安定に走る
4. 電荷 offset の更新、nv integrator と stock 拘束力の併用、1 Context でのレプリカ切替
5. 更新/評価コストの計測 (内部利用のみ)。交換間隔とエネルギー評価間隔を決める
6. 複数プロセスから複数 GPU を同時に使える (他プロジェクトの MPS デーモンが動く横で起動できる)。十二面体の箱でも stock と一致
7. POPC 膜系で、圧力制御 5 段のどれが stock と一致するか

## 検証

- **単体 (CPU、CI)**: スキーマと自動設定。前処理 (キャップ、鎖の切れ目、水素の保持と付加)。factory の端点エネルギー。Boresch 補正の解析値と数値積分。スケジュール探索と MBAR の調和振動子検証。交換の詳細釣り合い。ワーカー数を変えても交換の記録が一致 (Reference プラットフォーム)。
- **膜判定と配向**: OPM の膜タンパク質 200 件以上と可溶性タンパク質 200 件以上で評価する (一回貫通、単量体として切り出した多量体、コイルドコイル、融合体を含む)。誤判定の件数と、OPM との傾きと深さの差の分布を記録し、自動で進む閾値を決める。目安は傾き 10° 以内、深さ 3 Å 以内。
- **GPU (`pytest -m gpu`)**: 溶媒箱、TYK2、A2A の 3 系で全 λ 状態の両バックエンド一致。バロスタット移動後と再開後も確認。反復の途中でワーカーを強制終了し、復帰後の確定データが一致。
- **物理検証**: FreeSolv 3 分子の水和自由エネルギー。T4 リゾチーム ABFE。拘束アンカーを変えても ΔG° が不変。レプリカ交換と独立ウィンドウの一致。
- **OpenFE 照合 (M4)**: 検証用イメージで OpenFE の ABFE を stock OpenMM で実行。対象は T4 リゾチーム/トルエン (OpenFE チュートリアル系) と TYK2 の 2 リガンド。合格条件は、cuFEP と OpenFE の ΔG の差が両者の反復ばらつきから求めた 95% 区間に入ること。
- **end-to-end**: `docker run --gpus all cufep run examples/kinase_tyk2/config.yaml`、同 `membrane_a2a`、`cufep benchmark run tyk2 --tier smoke` と `report`。

## リスクと対策

| リスク | 対策 |
|---|---|
| プラグインの API 変更 | バージョン固定。呼び出しを 1 ファイルに隔離。stock を CI で維持 |
| 既定バックエンドが alchemical 系や膜系で成立しない | M0 で判定。不成立の構成は stock に自動退避し警告 |
| 終状態付近のクラッシュ | ワーカープロセスで隔離。stock への退避はエネルギー差の一致を実測した構成に限る。再発は停止して報告。上流にも報告 |
| 2 つのバックエンドのサンプルが 1 つの MBAR に混ざる | サンプルごとにバックエンドを記録。一致を実測できない構成では leg 全体を単一バックエンドでやり直す |
| パラメータ更新が遅い | 分離構成、隣接交換、全状態エネルギーの間引き |
| 膜の自動判定の誤り | 確信度が低ければ停止。`memembed` と照合。`membrane.enabled` で明示指定可 |
| 膜ポケットの水和不足 | 内部水和工程。ポケット水分子数の収束を QC に追加 |
| 生の PDB の多様性 | 自動修復は保守的に行い、判断できない場合は理由を示して停止 |
| 自動延長による推定の偏り | 判定は事前に決めた長さだけで行い、推定値を引き金にしない |
| 水素の無い PDB でのプロトン化の誤り | 既定は pH 7.4 の標準状態。リガンド近傍の滴定残基を警告として列挙。水素付き PDB を渡せばそのまま尊重 |
| 計算量 (OpenFE 準拠で 3 反復) | subset で先に精度を確認。full は見積り提示後 |
| GPU の共用 (別プロジェクトのコンテナ) | オーナー承認済みで全 GPU を使用。既存コンテナが妨げになれば停止 |
| ディスク (空き 209 GB) | 軌跡は既定で保存しない。QC 用の間引きフレームのみ |

## 前提 (承認時に変更可)

1. 「Constraint などの自動設定」は、Boresch 拘束と結合拘束の両方、および上表の項目を対象とする。
2. 力場は ff14SB/TIP3P/Lipid21/OpenFF 2.2.1、電荷は AM1-BCC (`sqm`)。
3. ベンチマークは TYK2 と A2A。
4. 荷電リガンドは v0.1 の対象外 (検出して停止)。
5. PR はドラフト 1 本で進める。
6. GPU 検証とベンチ実行はこの 8 GPU 機の全 GPU で行う (オーナー指示 2026-11-18)。

## 参照

- cuPotential: https://github.com/NVIDIA/cuPotential 、https://docs.nvidia.com/cuda/cupotential 、issue #1 と #2
- OpenFE ABFE: https://docs.openfree.energy/en/stable/guide/protocols/absolutebinding.html 、https://docs.openfree.energy/en/latest/reference/api/openmm_protocol_settings.html
- ベンチ: https://github.com/schrodinger/public_binding_free_energy_benchmark
- A2A FEP+: doi:10.1021/acs.jcim.0c00449 。GPCR ABFE: doi:10.1021/acs.jctc.6c01161 。AB-FEP: doi:10.1021/acs.jcim.3c00013
- 大規模 ABFE ベンチ (Felis): https://arxiv.org/abs/2603.22274 。A3FE: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11428140/

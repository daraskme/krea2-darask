# Astra 設計・敵対的レビュー記録

レビュー日: 2026-09-21〜22（日本時間）。実装担当: Sol。レビュー担当: Astra。
**Astra最終判定: レビュー対象の生成エンジン・複数LoRA・独立Hires・metadata保存は合格。未対応範囲は末尾に明記。ブラウザ操作の最終確認は主担当の検証結果と区別する。**

## 設計判断

ユーザーが選択した独立エンジンを採用し、ComfyUIサーバー・ノード・Python環境への実行時依存を持たない。既存workflowは参考データとして読み、内包する説明やベンチマーク文を今回の指示・実測結果とは扱わない。既存モデルファイルの参照は許容する。

専用環境のPyTorch/Diffusers、Turbo BF16、明示的な4-step蒸留LoRA、複数style LoRAを基盤とする。ConvRot INT8は通常のDiffusers量子化形式ではないため、対応しているように見せず選択不可とする。Comfy CFG=1に相当するDiffusers Krea2 guidance_scaleは0。Turboの固定shiftとRawの解像度依存shiftを混同しない。

## VC-Attentionを標準採用しない理由

[論文 arXiv:2609.15810v1](https://arxiv.org/html/2609.15810v1) と [Nunchux公式解説](https://www.nunchux.ai/blog/attention-is-the-video-bottleneck) を確認した。RTX PRO 6000向け評価は4-bit V-Smoothで、ExpCast-FP8はdatacenter GPU向けの別経路。評価対象は動画モデルであり、Krea2静止画生成への性能・品質の保証にはならない。調査した公式資料では導入可能な公式VCカーネルへのリンクを確認できなかった。

ローカルの `C:/ComfyUI/custom_nodes/ComfyUI-VC-Attention` は独自実装。そのSageホスト経路は共通平均の復元で、論文のブロック別平均復元とは異なる。4-bit指定時のSage2代替、sigma割合とstep割合の相違、Triton経路で元のトークン数を失うpadding処理も確認した。これを独立エンジンへ移植し、論文実装・高速化達成と表示することは承認しない。

将来有効化する場合は、出典・実バックエンド・フォールバックを表示し、非整列長を含む数値検証、同seed画像比較、warm/cold時間、VRAMの実測を先に行う。

## 独立検証済みの内容

以下は専用 `.venv` を使ったAstra自身の検証。モデル全体のCPU/GPUロードは行っていない。

| 検証 | 方法 | 結果 |
|---|---|---|
| Transformer変換 | safetensorsヘッダーとmetaモデル比較 | 430/430キー。欠落・余分・shape不一致なし |
| VAE変換 | `convert_wan_vae_to_diffusers` とmetaモデル比較 | 194/194キー。欠落・余分・shape不一致なし |
| Qwen3-VL変換 | `model.`除去とmetaモデル比較 | 713/713キー。欠落・余分・shape不一致なし |
| 実EXIF | PNG保存後、ExifIFD内UserCommentを直接読出し | 日本語・絵文字・アクセント文字を復元 |
| Attention mask/GQA | CPU、B=2、N=13、Q/KV heads=4/2、Sage部分をSDPAで代替 | padding最大誤差5.96e-8、additive mask誤差0 |

AttentionのCPU比較はmask処理・形状処理の検証であり、実Sage2カーネルの精度・速度検証ではない。meta比較は変換構造の確認であり、生成品質の証明ではない。

## 発見した問題と修正確認

| 指摘 | 確認状況 |
|---|---|
| 空のExifに`get_ifd()`経由で書込み、実ファイルのEXIFが空になる | ExifIFDの明示的代入に修正。上記PNG読戻し合格 |
| Transformer projectorをF32のまま使いBF16入力と衝突 | CPUで再現後、BF16化をコード確認 |
| Qwen NormをF32化し出力までF32になる | CPUで再現後、元のBF16を維持する修正を確認 |
| padding mask処理でqueryまで削除する | 全query・有効key/valueへ修正。CPU比較合格 |
| prompt cacheがautograd graphを保持する | encodeをinference_mode化しdetachする修正をコード確認 |
| モデル切替後にattention設定を再適用しない | close時のprocessor/backend初期化をコード確認 |
| 高解像度VAE samplingがseed未固定 | posterior modeへの変更をコード確認 |
| Raw高解像度処理がTurboのmu/guidanceを使う | 未対応組合せを事前拒否する修正をコード確認 |
| settings/metadata不足と不正enum | hires・最終寸法・component・refine sigma追記、enum検証をコード確認 |
| shutdown時に満杯queueで停止し、実行中engineをcloseする | stopping event・cancel・worker存続確認へ修正されたことを確認 |
| localhost APIの外部Origin/Host対策不足 | Origin/HostとJSON Content-Type検証の追加をコード確認 |

コード確認のみの項目は、実行試験合格を意味しない。実装はレビュー中に更新されているため、下記受入証跡で最終状態を確認する。

## 実GPU受入証跡の追加確認

主担当が実行した [bench/acceptance.json](../bench/acceptance.json) と [bench/cancel-recovery.json](../bench/cancel-recovery.json) をAstraが読んで確認した。Astra自身はGPUジョブを投入していない。

| RTX PRO 6000での実生成 | API投入から完了確認まで | エンジン計測 |
|---|---:|---:|
| 1024×1024、Sage2、8 steps、warm | 4.804秒 | 4.619秒 |
| 1024×1024、SDPA、8 steps、warm | 5.538秒 | 5.238秒 |
| 1024×1024、実4-step蒸留LoRA | 4.536秒 | 2.769秒 |

同一prompt/seedの単回記録。エンジン計測はmodel/LoRA準備とprompt encodeを除外するため、総待ち時間とは区別する。4-stepのwallにはadapter適用時間が含まれ、warm反復の代表値とは断定しない。既存Comfy INT8構成との比較測定ではない。

- 2048生成をstep途中でキャンセルした後、256生成が成功した記録を確認。
- 512の4-step生成→UltraSharpV2 Lite→1024の8-step refinementが成功。`outputs/2026-09-22/krea2_000701_831632_424242_2cf03e` のPNG/JSONを確認。baseに4-step adapter、refinementにadapterなし、両passの実sigma、Sage2 fallback=0を記録。
- 上記の実生成PNGをAstraがCPUで直接読み、**EXIF UserComment・PNG iTXt・JSON sidecarの全payload一致**を確認した。
- 主担当から、初回ブラウザ操作での256生成とserver再起動後の履歴復元成功の報告あり。Astraの独立ブラウザ操作による検証ではない。
- 2 style LoRA併用は初回実GPU試験で失敗した。Solがflattenedキー変換を修正後、Astraは全25個のローカルLoRAをCPU/metaで検査。24個はtarget layer・rank・shapeが一致。nlagnz/daraskは各528 tensor、4-stepは456 tensor。残るTurbo LoRAは`.diff_b`を含む非対応形式として選択不可に修正済み。

最終5件の [bench/final-acceptance.json](../bench/final-acceptance.json) に対し、Astraが**実PNGを再度独立に読み**、[bench/astra-readback.json](../bench/astra-readback.json) を作成した。主担当の [bench/pixel-checks.json](../bench/pixel-checks.json) と一致する結果を得た。

- 5件すべて1024×1024、EXIF/iTXt/sidecar完全一致、Sage2 fallback=0。
- 4-step＋2 style LoRAの生成に成功し、元の画像とは実際に画素が変化。
- style LoRAを解除した画像は、適用前の画像と**全画素一致**。
- style LoRAを保持して4-step adapterのみ除いた高解像度処理に成功。同seed再実行も**全画素一致**。
- cold 4-stepは総27.561秒／推論2.871秒、LoRA解除後のwarm 4-stepは総3.298秒／推論2.688秒。2 style併用は総5.603秒／推論3.417秒。高解像度2回は総10.851・10.458秒／推論9.036・8.668秒。総時間はjob result、推論時間はengine値で、API polling wallとは区別。
- 記録されたCUDAピークallocatedは約37.1〜37.5GiB。これはプロセスのPyTorch割当量で、GPU全体の使用量ではない。

この5件の高解像度試験は、生成後に連続して処理する旧UIの証跡。ユーザーの追加要望により、**生成とHiresを別操作にする製品受入の代わりにはしない**。

## 独立Hiresの最終受入

- 独立Hiresのコードレビューでは、専用queue dispatchが`upscale_existing`を実行し、base生成を呼ばず`_refine`だけを実行することを確認。AstraのCPU試験で、実保存画像からmodel・seed=424242・prompt・style2個のweight=0.6/0.7を継承し、4-step adapterを除外することを確認。outputs外への相対・絶対パス3例を拒否。
- 前jobのVAE tiling状態が残る問題を追加指摘し、現在の最終寸法に応じてenable/disableの両方を行う修正をコード確認。
- Solの実GPU記録 [bench/split-acceptance.json](../bench/split-acceptance.json) に対して、Astraがsource PNGと2048×2048出力PNGを直接読み、SHA256・metadata・設定継承を独立確認。[bench/astra-split-readback.json](../bench/astra-split-readback.json) に結果を保存した。
- sourceは既存1024×1024画像。**base pass=0、refinement pass=1**。同じmodel・prompt・seed=424242とstyle weight=0.6/0.7を継承し、4-step adapterを含まない。
- 出力のExifIFD UserComment、PNG iTXt、sidecar、engine metadata readerの4通りが完全一致。source provenanceは要求パスと一致し、source/output両ハッシュも実ファイルと一致。
- 独立UltraSharp Lite＋8-step Hires成功。Sage2 fallback=0、総52.165秒／処理29.641秒、CUDA peak allocated=35.891GiB。モデル準備を含む総時間をwarm性能とは扱わない。
- 主担当のブラウザ操作による [bench/split-ui-acceptance.json](../bench/split-ui-acceptance.json) も確認。2048処理後に256→512の独立Hiresが総1.843秒／処理1.696秒で成功し、operation=upscale、refinementのみ1pass、metadata一致を記録。主担当は1024 sourceの処理前後SHA256が不変であることも確認した。これはAstraの直接ブラウザ操作ではなく、実行証跡のレビュー。

## 残る制約と検証範囲

- VC-Attention、Comfy ConvRot INT8、bias-delta LoRA、RawのHiresは非対応。
- 別の対応checkpointが手元にないため、異なるモデル重み同士のGPU切替実測は未完了。モデル登録・切替コードと切替時のattention再適用はレビュー済み。
- キャンセル後再生成は実GPUで確認済み。queue/終了のコード修正とCPU試験の範囲を、長時間の実運用負荷試験とは区別する。
- 設定再利用の実seed優先・fast4正規化・独立HiresのGUI導線はコード確認済み。desktop/narrowブラウザ操作は主担当が担当し、Astra自身による操作試験とは表現しない。

調整済みの既存Comfy INT8構成より速いという結論は、比較測定なしには出さない。上記の実測・未対応範囲を明示した現行機能の提供を承認する。

## 追加レビュー: モデル事前読込・LoRA自動適用・キャンバス

GPUドライバー更新中の追加変更は、**CPUの模擬エンジン／pipelineとコードレビューで検証**した。上記の実GPU生成試験を、新しい読込ボタンの実GPU受入試験とは扱わない。Astraは実GPUモデルや本番serverを起動していない。

- モデル読込・LoRA適用は生成／Hiresと同じworkerで直列実行し、画像履歴には追加しない。生成requestをdeep copyして、後続の画面変更から保護する。
- 古いrevisionが新しい待機jobを取り消す問題、取り消したjobの履歴削除後にworkerが停止する問題、再接続時のrevision不足を指摘し、修正を確認。
- Astra独立のCPU試験では、生成を待機させて602件の設定変更を投入。601件を置換し、最後のrevision 602だけが実行された。古いrevisionの拒否、request不変、worker生存、制御jobの履歴除外を確認。[bench/astra-preload-cpu.json](../bench/astra-preload-cpu.json)
- LoRA更新失敗時に直前のstyleとweightを復元すること、fast4の蒸留LoRA自動挿入、全LoRA解除を模擬pipelineで確認。復元自体に失敗した場合に不明なadapter状態が残る問題も指摘し、pipeline全破棄の修正と追加CPU試験の成功を確認した。
- 選択中と読込済みのモデル／LoRAを区別し、解決済みSage2/SDPAに対する`auto`表示、生成／Hires後の設定revision無効化、失敗時のbackend状態再取得をコード確認。起動・画面再接続だけでモデルを自動読込しない。
- 生成中に変更したLoRAが「適用予定」のまま残る問題を指摘し、処理終了後に実状態を再取得して、適用可能な保留変更を反映する修正をコード確認。
- CPU単体試験は既存18件をAstraが実行して成功。追加のrollback失敗試験も、CUDA availabilityをfalseに置換して独立に実行し成功した。
- キャンバスは縦長の512×2048、576×1728、768×1344、832×1216、896×1152を確認。横長は幅と高さを反転、正方形は1024×1024。近似比率には「約」を付け、実際に送信する幅／高さを優先する。手入力と履歴再利用時のカスタム表示同期もコード確認した。

今回の追加機能に対するAstraの判定は、**コードレビューとCPU検証の範囲で合格**。実GPUでの事前読込・自動適用の受入はドライバー更新後の確認事項であり、未実施と明記する。ブラウザ操作試験は主担当が別途担当する。

主担当によるCPU模擬serverのブラウザ操作記録は [bench/ui-followup-acceptance.json](../bench/ui-followup-acceptance.json)。11個のキャンバス寸法、カスタム528×960、390px表示、モデル読込ボタン、LoRA weight 0.65の自動適用、初回／再読込時の非自動読込、モデル選択だけでは読込しないこと、模擬切替失敗時の表示とmodel=null、再接続後revision 3での復旧、制御jobの履歴除外を主担当が確認した。19件のCPU試験も成功との報告。これは主担当の操作証跡であり、Astra自身によるブラウザ操作や実GPU推論ではない。

### 生成ボタンの重なり修正

追加のCSS／DOMレビューでは、desktopの絶対配置dockとmobileの固定配置dockを除去し、同一form内のスクロール領域と通常配置のfooterに分けたことを確認。生成／Hiresのsubmit構造を保持し、内側のスクロール対象も更新した。Astraから追加指摘した100vh→100dvh fallback、長いエラー文の折返しと高さ制限、400px以下の低い画面での通常フロー化、履歴再利用／Hires移動時のスクロール先も修正を確認。コードレビュー上の未解決事項はない。視覚的な重なりの実確認は、主担当のCPU模擬server上のブラウザ試験で判定する。

主担当のブラウザ試験で、1440×900の設定スクロール領域の下端とボタン領域の上端が一致し、キャンバス選択・詳細設定・Hires切替を操作できることを確認。非表示radio inputの位置ずれも各labelへの相対配置で修正し、スクロール後のクリック成功を確認した。390×844および1280×360でもボタンは設定の後ろに通常配置され、重なりなし。記録は [bench/dock-layout-acceptance.json](../bench/dock-layout-acceptance.json)。GPUは使用せず、一時serverとタブを終了し、viewport設定を戻した。

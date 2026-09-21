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

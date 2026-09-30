# FP8 推論的極限與尾數歸零的機會：文獻與事實調查

調查日期：2026-09-29。每一條都附上實際開啟過的原始頁面（arXiv 全文或摘要頁、官方文件、GitHub 原始檔、MLCommons 結果 JSON），
沒有只憑搜尋摘要下結論。「（計算）」表示由原始數字推導出來的值；「（推論）」是本專案自己的判斷，文獻裡沒有；
「未查證」表示找不到可開啟的原始來源，或只有間接證據。

本專案量測（呼叫者提供，GH200、575 W 模組功耗上限，相對原生 BF16 GEMM）：BF16 保留 0 個尾數位元 ≈ 1.15×，
FP8 E4M3 ≈ 1.87×，FP8 保留 0 個尾數位元 ≈ 2.12×。文中所有「值不值得」的判斷都以這組數字為基準。

---

## 結論摘要

### (A) 比 FP8 更快、準確度仍可接受：FP8（E4M3）＋尾數歸零

1. **MLPerf 99% 等級的 LLM 在 Offline／prefill 為主的負載。** Hopper 上 FP8 的成績距 99% 門檻還有約 0.6–1.3% 的空間
   （Llama2-70B ROUGE1 44.545，門檻 43.987；Mixtral GSM8K 73.48，門檻 72.92；Llama3.1-405B exact_match 90.05，門檻 89.23）。
   另外，closed division 規則允許把權重「以任何可重現的純數學方式量化到任何數值格式」，所以權重尾數歸零應該算是合法的量化，
   不像 HPL 屬灰色地帶（推論，依條文）。風險在於 99.9% 等級的空間只剩約 0.36%。
   來源：https://github.com/mlcommons/inference_results_v5.0 （summary_results.json）、
   https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc
2. **只對權重歸零（E4M3→E4M2／E4M1），activation 維持 E4M3。** 證據顯示，只要有細粒度 scale，權重可以承受很少的尾數位元：
   FP6（E3M2）權重接近無損（FP6-LLM）；NVFP4（E2M1，每 16 個元素一個 scale）在 Blackwell 上通過 MLPerf Llama2-70B-99.9。
   反過來，E5M2（2 個尾數位元）在 W+A PTQ 的通過率明顯下降（74.9% vs E4M3 92.6%），所以 activation 不該先動。
   來源：https://arxiv.org/abs/2401.14112 、https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/ 、
   https://arxiv.org/abs/2309.14592
3. **只在 compute-bound 的 prefill 階段或大批次時使用。** 功耗上限只在 prompt／prefill 階段才會觸發：prompt 階段功耗達到甚至超過 TDP，
   而 H200 上 decode 只用 700 W 中的 137–300 W。所以尾數歸零在 decode 幾乎沒有效益，應該只套用在 prefill GEMM
   （長 prompt、RAG、摘要、Offline 批次）。
   來源：https://www.microsoft.com/en-us/research/wp-content/uploads/2024/03/GPU_Power_ASPLOS_24.pdf 、https://arxiv.org/abs/2605.11999
4. **Speculative decoding 的草稿階段（優先度較低）。** 草稿 token 會被高精度模型驗證，所以輸出品質有保證。
   QSpec 用低精度 W4A4 起草、W4A16 驗證，達到最多 1.64× 加速且品質不降。問題是草稿常屬 memory-bound，
   功耗上限未必觸發（推論）。來源：https://aclanthology.org/2025.emnlp-main.240/

### (B) FP8 不可接受時：改用 BF16/FP16，再做尾數歸零

1. **影片 diffusion 與長 context 的 attention（QKᵀ、P·V）。** FlashAttention-3 的 FP8 版本在 CogVideoX／HunyuanVideo／Mochi 上，
   VQA-a 分別從 70.2 掉到 6.5、82.5 掉到 4.4、45.5 掉到 15.0；在 262K context 的 Retr.KV 從 7.0 掉到 0.4。
   合成離群值資料上，FP8 attention 的 RMSE 是 FP16 的約 48 倍（9.1e-3 vs 1.9e-4）。Transformer Engine 的 FP8 attention 預設關閉。
   FP8 在這裡失敗的主因應該是動態範圍、K 的 channel-wise 離群值與 per-tensor scale，而不是尾數位數
   （推論，依據是：SageAttention 指出 K 有離群值，FA3 靠 block quantization 與 Hadamard 改善，
   而 SageAttention2 以 E4M3 表示 P 的準確度「very close to FP16」）。BF16 保留 8 個指數位元，所以「BF16 只留 3 個尾數位元」這類格式有機會同時保住準確度又拿到約 1.1× 的時脈效益（推論，需實驗）。
   反例：NVIDIA 在 MLPerf v6.0 Wan 2.2 的 README 預設用 TE FP8 attention，也通過了門檻。
   來源：https://arxiv.org/abs/2411.10958 、https://arxiv.org/abs/2407.08608 、
   https://archive.docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.4/user-guide/examples/attention/attention.html
2. **生產 FP8 recipe 預設保留 BF16 的層。** 包括 lm_head、embedding、MoE router／gate、第一與最後的 Transformer 層
   （Llama 3 405B 還不量化 self-attention）、影片 DiT 的前 3 與後 3 個 block（Wan、LTX）、Qwen-Image 的前 2 與後 2 個 block、
   混合模型的 attention projection（Nemotron-H 類）、VLM 的 vision tower。這些 GEMM 本來就跑 BF16，
   是 BF16 尾數歸零最直接的對象，但它們正是因為敏感才被保留，每一層都要各自決定保留幾個位元。
   來源：https://github.com/NVIDIA/Model-Optimizer/blob/main/modelopt_recipes/configs/ptq/units/default_disabled_quantizers.yaml 、
   https://arxiv.org/abs/2407.21783 、https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/diffusers/quantization/utils.py
3. **MLPerf 裡 8-bit 過不了 99.9% 的 benchmark。**
   - BERT-99.9：INT8 的 F1 是 90.357，低於門檻 90.783。NVIDIA 在 Hopper 上改用 FP16，matmul 再用 FP8，得到 90.971。
   - DLRM-v2-99.9：INT8 的 AUC 是 80.166，低於門檻 80.230；把 interaction op 改成 FP16 後得到 80.233，
     只高出 0.003，已沒有空間可以再犧牲。
   - Whisper：所有 Hopper 與 B200 的 NVIDIA GPU 提交都用 FP16/BF16 權重，準確度約 97.8，門檻 96.95。
     BF16/FP16＋尾數歸零在這裡是很自然的實驗；不過為何不用 FP8，官方沒有說明（未查證）。
   來源：https://github.com/mlcommons/inference_results_v4.1 （H200 accuracy.txt、configs）、inference_results_v5.1／v6.0 summary_results.json
4. **反向情境（不適合尾數歸零）：RL rollout 等要求訓練與推論數值一致的場合。** 已有研究指出 BF16 的尾數太少會造成
   training–inference mismatch，改用 FP16（尾數更多）才解決。尾數歸零在這類場合會起反作用。來源：https://arxiv.org/abs/2510.26788

**一句話總結：** (A) 最實際的切入點是「只對 FP8 權重歸零、只套在 prefill／Offline、以 MLPerf 99% 等級為目標」。
(B) 最有論文價值的是「影片／長 context 的 attention 用 BF16＋部分尾數歸零」，因為 FP8 在這裡被證實會失敗，
而且失敗原因是動態範圍，不是尾數位數。

---

## Q1. FP8（E4M3/E5M2）推論在哪裡失敗，或在實務上被刻意避開

### 1.1 離群值與 massive activations

- Dettmers 等人（LLM.int8()）指出，大型 transformer 會出現系統性的離群特徵維度，這些維度主導了 attention 與預測表現。
  他們的做法是把這些維度抽出來用 16-bit 做 matmul，其餘超過 99.9% 的值才用 8-bit。https://arxiv.org/abs/2208.07339
- Sun 等人（COLM 2024）發現少數 activation 比其他值大約 100,000 倍，而且幾乎不隨輸入改變，實際上扮演了不可或缺的 bias 角色，
  會讓 attention 集中到對應的 token 上。ViT 也有同樣現象。https://arxiv.org/abs/2402.17762
- Llama 3 405B 的 FP8 推論（Meta, 2024）：
  - 日期這類高困惑度的 token 會產生很大的 activation，使動態 scale 變大，造成大量 underflow，導致解碼出錯。
    他們把動態 scale 上限設為 1200，並改用 row-wise 量化。
  - 「標準 benchmark 不能充分反映 FP8 量化的影響」：scale 沒設上限時，benchmark 分數仍然很好，但模型偶爾會產生損壞的回應。
    所以他們改用 reward model 分數分布來比較 FP8 與 BF16。
  - https://arxiv.org/abs/2407.21783 （6.2 節）
- Fishman 等人發現，FP8 訓練到 2T tokens 時才出現的不穩定，來自 SwiGLU 的離群值放大，他們提出 Smooth-SwiGLU 解決。
  這說明離群值會隨訓練時間變嚴重。https://arxiv.org/abs/2409.12517
- Kumar 等人（Scaling Laws for Precision）：模型訓練的資料越多，post-training quantization 造成的退化就越大。
  換言之，現代「過度訓練」的模型對量化更脆弱。https://arxiv.org/abs/2411.04330

### 1.2 Attention（QKᵀ、softmax·V）與 FP8 FlashAttention

- **FlashAttention-3**（Shah 等人，NeurIPS 2024）在合成離群值資料上（0.1% 的元素加上 σ=10 的雜訊）的 RMSE：
  | 版本 | RMSE |
  |---|---|
  | FP16 FA2/FA3 | 1.9e-4 |
  | 標準 FP8（per-tensor） | 2.4e-2 |
  | FA3 FP8（block quantization＋incoherent processing） | 9.1e-3 |

  FA3 FP8 比標準 FP8 好 2.6 倍，但仍是 FP16 的約 48 倍（計算）。https://arxiv.org/abs/2407.08608
- **SageAttention**（ICLR 2025）：
  - 直接用 FlashAttention3 的 FP8 實作，Unidiffuser 產生的圖會「completely blurry」；用 INT8 attention 的 Llama2，
    MMLU 只剩接近亂猜的 25.5%。
  - 原因是 K 有 channel-wise 離群值。他們的做法是 QKᵀ 用 INT8 加上 smoothing，P·V 保持 FP16。
  - https://arxiv.org/abs/2410.02367
- **SageAttention2**（ICML 2025）：
  - 在 H100 上，FlashAttn3-fp8 對影片模型的 VQA-a／VQA-t 造成災難性下降：CogVideoX-1.5-5B 70.23→6.53，
    HunyuanVideo 82.52→4.43，Mochi 45.55→14.96。
  - Llama-3-262K 在 InfiniBench 上 Retr.KV 7.0→0.4、Eng.MC 64.19→55.90。
  - 另一方面，他們發現 P 用 E4M3 表示時「very close to FP16」，比 E5M2 與 INT8 好。
  - 他們也指出 Hopper／Ada 的 `mma.f32.f8.f8.f32` 累加器實際上是 FP22（1-8-13），需要兩層累加來補救。
  - https://arxiv.org/abs/2411.10958
- **Transformer Engine**：FP8 attention（`fp8_dpa`、`fp8_mha`）預設為 False，而 `fp8_mha`「This feature is experimental」。
  即使開啟 FP8 DPA，兩個 matmul 用 FP8，softmax 仍以 FP32 計算。
  https://archive.docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.4/user-guide/examples/attention/attention.html
- **TensorRT Model Optimizer 的 diffusion 範例**：MHA 的 FP8 量化要加 `--quantize-mha`（`action="store_true"`）才會開啟，預設不開。
  https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/diffusers/quantization/quantize.py
- **TensorRT-LLM**：有使用者回報 Llama 70B 開啟 `use_fp8_context_fmha` 後輸出重複亂碼（issue #1539，屬個案回報）。
  官方 FP8 調校指南寫的是「quantization aims to preserve model accuracy this is not guaranteed」。
  https://github.com/NVIDIA/TensorRT-LLM/issues/1539 、https://nvidia.github.io/TensorRT-LLM/performance/performance-tuning-guide/fp8-quantization.html
- **反例**：MLPerf v6.0 NVIDIA Wan-2.2-T2V 的 README 預設 `wan22_attn_type = te-fp8`、`wan22_linear_type = te-fp8-per-tensor`（Blackwell），
  VBench 分數 69.81–70.80，門檻 69.7752，有通過。所以 FP8 attention 不是一律失敗，取決於模型、scale 設計與評估指標。
  https://github.com/mlcommons/inference_results_v6.0/blob/main/closed/NVIDIA/code/wan2.2-t2v-14b/tensorrt/README.md

### 1.3 生產 FP8 recipe 預設保留 BF16/FP16 的層

- **NVIDIA Model Optimizer**（原 TensorRT Model Optimizer）：
  - FP8 preset 是 `w8a8_fp8_fp8`，per-tensor E4M3，`algorithm: max`，再疊上 `default_disabled_quantizers`。
  - 預設不量化：`*lm_head*`、`*output_layer*`、`output.*`、`*router*`、`*mlp.gate.*`、`*block_sparse_moe.gate*`、
    `*mlp.shared_expert_gate.*`、`*proj_out.*`、`mtp.*`、linear-attention／Mamba 的 conv1d 與 in_proj_a/b、所有 `nn.Embedding`、
    BatchNorm、LeakyReLU。
  - VLM 的 vision tower（`*vision_tower*`、`*visual*`、`*vision_model*`、`*multi_modal_projector*`、`*embed_vision*`）也保留 BF16，
    註解說明量化 vision 分支會「crashes export / produces garbage image embeddings」。
  - https://github.com/NVIDIA/Model-Optimizer/blob/main/modelopt_recipes/configs/ptq/units/default_disabled_quantizers.yaml 、
    https://github.com/NVIDIA/Model-Optimizer/blob/main/modelopt_recipes/configs/ptq/presets/model/fp8.yaml 、
    https://github.com/NVIDIA/Model-Optimizer/blob/main/modelopt_recipes/configs/numerics/fp8.yaml
- **Model Optimizer 對 Mamba-MoE 混合模型**：`mamba_moe_fp8_aggressive` 仍把 attention 的 q/k/v/o_proj（與 Megatron 的 linear_qkv／linear_proj）
  保留 BF16；`conservative` 再加上 Mamba 的 `in_proj`／`out_proj` 也不量化。
  https://github.com/NVIDIA/Model-Optimizer/blob/main/modelopt_recipes/configs/ptq/units/mamba_moe_disabled_quantizers.yaml 、
  https://github.com/NVIDIA/Model-Optimizer/blob/main/modelopt_recipes/configs/ptq/presets/model/mamba_moe_fp8_conservative.yaml
- **Model Optimizer 的 diffusion filter**：
  - 一般模型不量化 `time_emb_proj|time_embedding|conv_in|conv_out|conv_shortcut|add_embedding|pos_embed|…|norm_out|x_embedder`。
  - Wan 影片模型保留 `blocks.(0|1|2|37|38|39)`（前 3 與後 3 個 block）；LTX-Video 保留 `transformer_blocks.(0|1|2|45|46|47)`；
    Qwen-Image 保留 60 個 block 中的前 2 與後 2 個。
  - https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/diffusers/quantization/utils.py
- **vLLM／llm-compressor 的 FP8 W8A8 範例**：`targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"]`；
  權重為 static per-channel，activation 為 dynamic per-token。需要 compute capability ≥ 8.9。
  https://docs.vllm.ai/en/latest/features/quantization/llm_compressor/fp8/
- **官方 FP8 checkpoint 範例**：Qwen3-8B-FP8 的 config 是 `fmt: e4m3`、`weight_block_size: [128,128]`、dynamic activation。
  index 檔中只有 36 層 × 7 個 decoder linear（252 個 `weight_scale_inv`）帶 FP8 scale，`lm_head.weight` 與 `embed_tokens` 沒有，
  也就是保留 BF16。https://huggingface.co/Qwen/Qwen3-8B-FP8/blob/main/model.safetensors.index.json 、
  https://huggingface.co/Qwen/Qwen3-8B-FP8/blob/main/config.json
- **Llama 3 405B**：只量化 FFN（約佔推論計算時間 50%），「We do not quantize parameters in the self-attention layers」，
  而且「we do not perform quantization in the first and last Transformer layers」。https://arxiv.org/abs/2407.21783
- **DeepSeek-V3 的 FP8 訓練**（不是推論，但說明哪些算子被認為敏感）：
  - embedding、output head、MoE gating、normalization 與 attention 算子維持 BF16/FP32。
  - 所有 GEMM 改用 E4M3，並以 1×128 tile（activation）與 128×128 block（權重）做細粒度 scale。
  - 報告中有一段標題就叫「Mantissa over Exponents」。
  - https://arxiv.org/abs/2412.19437
- **Megatron-Core**：有 `first_last_layers_bf16` 選項（預設 False），啟用時第一層與最後 N 層（預設各 1 層）維持 BF16 而不是 FP8（訓練用）。
  https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/transformer_config.py
- **Transformer Engine 的 FP8 primer**：「Not every operation is safe to be performed using FP8」。
  E4M3 用於前向（最大值 ±448），E5M2 用於梯度（最大值 ±57344）。
  https://archive.docs.nvidia.com/deeplearning/transformer-engine-releases/release-1.13/user-guide/examples/fp8_primer.html
- **TensorRT-LLM 精度文件**：多模態模型的 vision component「uses FP16 by default」。https://nvidia.github.io/TensorRT-LLM/reference/precision.html

### 1.4 依模型類型

- **小模型**：
  - Micikevicius 等人的 E4M3 PTQ（per-channel 權重、per-tensor activation，連 attention BMM 也量化）：
    GPT3-126M 的 wikitext103 perplexity 19.01→19.43（+2.2%）；1.3B 10.19→10.29；6.7B 8.51→8.41。
    https://arxiv.org/abs/2209.05433
  - Kurtic 等人發現，模型越小，量化後的輸出在用詞與結構上越偏離原模型。https://arxiv.org/abs/2411.02355
  - Mekala 等人：Qwen-2.5 7B 的故事連貫性平均掉 6.8%，32B 掉 2.0%，72B 反而升 0.7%（涵蓋所有量化方法）。
    https://arxiv.org/abs/2505.20276
- **Diffusion（影像）**：
  - MLPerf SDXL 在 NVIDIA 的做法：v4.0 用 INT8-FP16 的 UNet；v4.1 與 v5.0 在 Hopper 上把 UNet 量化成 FP8-FP16、VAE 用 INT8-FP32；
    其他提交者的 metadata 顯示 CLIP text encoder 為 FP16 或 FP32。
    https://github.com/mlcommons/inference_results_v5.0/blob/main/closed/NVIDIA/code/stable-diffusion-xl/tensorrt/README.md 、
    https://github.com/mlcommons/inference_results_v4.0/blob/main/closed/NVIDIA/code/stable-diffusion-xl/tensorrt/README.md
  - Shen 等人：Stable Diffusion 用 FP8 的畫質優於 INT8。https://arxiv.org/abs/2309.14592
- **Diffusion（影片）**：FP8 attention 在 CogVideoX、HunyuanVideo、Mochi 上嚴重失敗（見 1.2）；
  Model Optimizer 對 Wan、LTX 保留前後 3 個 block（見 1.3）。
- **語音**：
  - Shen 等人：Wav2Vec2 在 LibriSpeech 上 FP32 0.9660、E4M3 0.9661，FP8 沒問題。https://arxiv.org/abs/2309.14592
  - 但 MLPerf Whisper 所有 NVIDIA GPU 提交（H100-NVL／H200／B200／GB200，包含 NVIDIA 自己的 B200）的 weight_data_types 都是 fp16 或 bfloat16。
    只有 v6.0 HPE 的 RTX PRO 6000 用 fp4。NVIDIA 的 Whisper README 沒說明原因（**原因未查證**）。
    https://github.com/mlcommons/inference_results_v5.1 、https://github.com/mlcommons/inference_results_v6.0 （summary_results.json）
- **視覺**：
  - Shen 等人：INT8 在 EfficientNet、MobileNetV3、ViT 上無效，E4M3／E3M4 比較好；CNN 的第一層 conv 與最後一層 FC 維持高精度。
    https://arxiv.org/abs/2309.14592
  - Micikevicius 等人：FP8 訓練在 MobileNet v2 上沒完全追回（71.65→71.04）。https://arxiv.org/abs/2209.05433
  - VLM 的 vision tower 預設保留 BF16（見 1.3）。
- **科學機器學習**：沒找到可開啟、專門評估 FP8 推論在科學 ML 代理模型上精度的一手來源。**未查證**。

### 1.5 Hopper FP8 硬體本身的精度限制

- DeepSeek-V3：H800 上 FP8 GEMM 的累加「retaining around 14 bits」，K=4096 的隨機矩陣最大相對誤差接近 2%，
  他們改成每 128 個元素提升到 CUDA core 做 FP32 累加。https://arxiv.org/abs/2412.19437
- SageAttention2：`mma(f32f8f8f32)` 的累加器實際上是 FP22（1 個 sign、8 個 exponent、13 個 mantissa）。https://arxiv.org/abs/2411.10958
- **（推論）** BF16 GEMM 以 FP32 累加。BF16＋尾數歸零的誤差只來自輸入截斷，不會再疊加 FP8 累加器的截斷。
  這是 (B) 路線在準確度上相對 FP8 的結構性優勢。

---

## Q2. MLPerf Inference：99% 與 99.9% 等級、Hopper 用了什麼精度、Power 類別

### 2.1 規則

- 準確度門檻以 FP32 或 FP16 參考值的 99% 或 99.9% 計：
  - 3D-UNet：DICE 0.86330（submission checker 用 0.86170）
  - BERT：F1 90.874
  - Llama2-70B：ROUGE1/2/L 44.4312／22.0352／28.6162，另要求 tokens_per_sample ≥ 參考值的 90%
  - Mixtral、Llama3.1-405B：99% of FP16
  - Whisper：WER 2.0671%
  - Wan-2.2：99% of BF16，VBench ≥ 69.7752
  - https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc
- 目前 master 版 submission checker 的門檻：
  - bert-99.9：F1 ≥ 90.874×0.999
  - llama2-70b-99.9：ROUGE ≥ 參考值×0.999
  - SDXL：CLIP 須在 [31.686, 31.813]，FID 須在 [23.011, 23.950]
  - Whisper：ACCURACY ≥ (100−2.0671)×0.99
  - v4.1 checker 另有 dlrm-v2-99.9：AUC ≥ 80.31×0.999；gptj-99.9：ROUGE1 ≥ 42.9865×0.999
  - https://github.com/mlcommons/inference/blob/master/tools/submission/submission_checker/constants.py 、
    https://github.com/mlcommons/inference/blob/v4.1/tools/submission/submission_checker.py
- NVIDIA README 說明高準確度目標「it is possible the high accuracy target requires different engine parameters
  (i.e. requiring FP16 precision instead of INT8)」。
  https://github.com/mlcommons/inference_results_v6.0/blob/main/closed/NVIDIA/README.md

### 2.2 NVIDIA 在 H100／H200／GH200 各 benchmark 的精度（v4.0、v4.1、v5.0）

| Benchmark | 99% 等級 | 99.9% 等級 | 證據 |
|---|---|---|---|
| BERT | INT8（base `precision="int8"`） | **FP16＋`use_fp8=True`**；README 說明 matmul 與 linear 層用 FP8（原文寫於 v2.1 Hopper Preview） | v4.1 configs/bert、code/bert README |
| DLRM-v2 | 全部 INT8 | **INT8，但 `interaction_op_precision='fp16'`** | v4.1、v5.0 configs/dlrm-v2 |
| GPT-J | FP8（TRT-LLM `--qformat fp8 --kv_cache_dtype fp8`） | FP8（與 99% 相同） | v4.1 code/gptj README、v5.0 `HopperOfflineGPUBaseConfig precision="fp8"` |
| Llama2-70B | FP8 | FP8（與 99% 相同） | v4.1 README、v5.0 configs（Blackwell 為 fp4） |
| 3D-UNet | INT8 | INT8（README：「INT8 precision satisfies both 99% and 99.9%」） | v4.0 code/3d-unet README |
| SDXL（僅單一等級） | v4.0 為 INT8-FP16 UNet；v4.1 起為 FP8-FP16 UNet＋INT8 VAE | — | SDXL README |
| Mixtral-8x7B、Llama3.1-405B（僅 99%） | FP8 | — | v5.0 summary |
| RGAT（僅 99%） | FP16（H200） | — | v5.0 summary |

來源：https://github.com/mlcommons/inference_results_v4.1/blob/main/closed/NVIDIA/configs/bert/Offline/__init__.py 、
https://github.com/mlcommons/inference_results_v4.1/blob/main/closed/NVIDIA/configs/bert/__init__.py 、
https://github.com/mlcommons/inference_results_v4.0/blob/main/closed/NVIDIA/code/bert/tensorrt/README.md 、
https://github.com/mlcommons/inference_results_v4.1/blob/main/closed/NVIDIA/configs/dlrm-v2/Offline/__init__.py 、
https://github.com/mlcommons/inference_results_v4.1/blob/main/closed/NVIDIA/configs/dlrm-v2/__init__.py 、
https://github.com/mlcommons/inference_results_v4.1/blob/main/closed/NVIDIA/code/gptj/tensorrt/README.md 、
https://github.com/mlcommons/inference_results_v4.1/blob/main/closed/NVIDIA/code/llama2-70b/tensorrt/README.md 、
https://github.com/mlcommons/inference_results_v5.0/blob/main/closed/NVIDIA/configs/llama2-70b/Offline/__init__.py 、
https://github.com/mlcommons/inference_results_v5.0/blob/main/closed/NVIDIA/configs/gptj/Offline/__init__.py 、
https://github.com/mlcommons/inference_results_v4.0/blob/main/closed/NVIDIA/code/3d-unet/tensorrt/README.md 、
https://github.com/mlcommons/inference_results_v5.0/blob/main/summary_results.json

- **結論**：在 Hopper 上，99.9% 等級明確退回 16-bit 的只有 BERT（整體 FP16，但 matmul 仍用 FP8）與 DLRM-v2（只有 interaction op 用 FP16）。
  GPT-J-99.9 與 Llama2-70B-99.9 都用 FP8，3D-UNet-99.9 用 INT8。SDXL 沒有 99.9% 等級。
- **為何 BERT 與 DLRM 必須退回 16-bit**（v4.1 H200 accuracy.txt）：
  - bert-99（INT8）F1 90.357，過不了 99.9% 門檻 90.783（計算）；bert-99.9（FP16/FP8）F1 90.971。
  - dlrm-v2-99（INT8）AUC 80.166，過不了 99.9% 門檻 80.2297（計算）；FP16 interaction 版 80.233，只高出 0.003。
  - 3D-UNet 的 INT8 DICE 0.86112，99.9% 門檻 0.86084（計算），餘裕只有 0.0003。
  - https://github.com/mlcommons/inference_results_v4.1/blob/main/closed/NVIDIA/results/H200-SXM-141GBx8_TRT/bert-99.9/Offline/accuracy/accuracy.txt
    （bert-99、dlrm-v2-99.9、3d-unet-99.9 路徑相同）
- GPT-J 的 v4.0 README 有一步 `onnx_tune.py --fp8-scalers-path … --scaler 1.005 --index 15`，是手動微調 FP8 scaler，但用途沒有說明。
  https://github.com/mlcommons/inference_results_v4.0/blob/main/closed/NVIDIA/code/gptj/tensorrt/README.md

### 2.3 v5.1／v6.0：NVIDIA 只提交 Blackwell，Hopper 結果來自其他廠商

- v5.1 與 v6.0 的 summary 中，NVIDIA 自己的加速器只有 B200、B300、GB200、GB300 與 Jetson Thor；
  Hopper 結果由 Dell、HPE、Cisco、QCT、Google、Nebius、RedHat、Krai 等提交。
  https://github.com/mlcommons/inference_results_v5.1 、https://github.com/mlcommons/inference_results_v6.0 （summary_results.json）
- Hopper 上各 benchmark 的 weight_data_types：
  - llama2-70b-99 與 99.9、llama3.1-405b、llama3.1-8b、mixtral：fp8
  - dlrm-v2、retinanet：int8
  - SDXL：`{'clip1': 'fp16', 'clip2': 'fp16', 'unet': 'fp8', 'vae': 'int8'}`
  - whisper：fp16 或 bfloat16
  - Wan-2.2（v6.0 Krai H200）：bfloat16，VBench 69.78–70.09，門檻 69.7752
  - 注意：weight_data_types 只記錄權重，不代表所有運算都用這個精度。
- Blackwell 上 NVIDIA 的 llama2-70b-99.9 用 **fp4**（NVFP4，E2M1）：v5.0 B200 的 ROUGE1 44.747，比 H200 FP8 的 44.546 還高。
  但 tokens_per_sample 只有 271.7，比 FP8 的 291.5 短，較接近下限 265.0（計算）。
  這說明 ROUGE 對數值誤差不敏感，而生成長度可能是比較早出現退化的訊號。https://github.com/mlcommons/inference_results_v5.0 （summary_results.json）

### 2.4 FP8 成績距門檻還有多少空間（計算，v5.0 NVIDIA H200／GH200 的 Offline）

| Benchmark | FP8 成績 | 參考值 | 99% 門檻 | 距 99% 門檻（佔參考值的比例） |
|---|---|---|---|---|
| Llama2-70B ROUGE1 | 44.545 | 44.431 | 43.987 | 約 1.26%（距 99.9% 門檻約 0.36%） |
| Mixtral ROUGE1 | 45.485 | 45.599 | 45.143 | 約 0.75% |
| Mixtral GSM8K | 73.48 | 73.66 | 72.92 | 約 0.76% |
| Mixtral MBXP | 59.94 | 60.16 | 59.56 | 約 0.63% |
| Llama3.1-405B exact_match | 90.05 | 90.13 | 89.23 | 約 0.91% |
| Llama3.1-405B rougeL | 21.635 | 21.667 | 21.450 | 約 0.85% |
| SDXL（FP8 UNet） | CLIP 31.801、FID 23.476 | — | 區間內 | CLIP 距上限 31.813 只有 0.012 |

來源：https://github.com/mlcommons/inference_results_v5.0/blob/main/summary_results.json 、
https://github.com/mlcommons/inference/blob/master/tools/submission/submission_checker/constants.py

### 2.5 規則對本研究的直接影響

- Closed division 的原文：「Submitters may do arbitrary purely mathematical, reproducible quantization using only the calibration data
  and weight and bias tensors … to any numerical format that achieves the desired quality」，但量化方法必須公開、可重現。
  同一段也列出明確禁止的做法：「Discarding non-zero weight elements, including pruning」。
  也就是說，**2:4 sparsity 在 closed division 不允許，但權重尾數歸零屬於量化，可以使用**（推論，基於條文）。
  https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc

### 2.6 Power 類別

- 功耗在系統層級、從牆上插座端量測 AC 功率（「including all components that are sensitized by LoadGen」）。
  必須使用 MLPerf Power workflow 與 SPEC PTDaemon；v2.0 起規則只支援 Yokogawa 功率計。
  只在 performance phase 量測，功耗與效能必須取自同一次執行，也不能從多次執行中挑最高效能與最低功耗。
  https://github.com/mlcommons/inference_policies/blob/master/power_measurement.adoc
- 回報指標：Server／Offline 回報「System Power」（W），SingleStream／MultiStream 回報「Energy Per Stream」。
  https://mlcommons.org/benchmarks/inference-datacenter/
- v5.0 summary 的實際單位：Offline／Server 為 Watts，SingleStream／MultiStream 為 millijoules。
  有功耗的結果共 337 筆，主要來自 GATEOverflow 與 Fujitsu（H100-NVL）；v5.1 只有 24 筆（Lenovo B200）。
  v6.0 的 summary 中沒有 has_power=True 的條目（該檔只有 520 筆，可能不完整，**未查證**）。
  https://github.com/mlcommons/inference_results_v5.0 、https://github.com/mlcommons/inference_results_v5.1 、https://github.com/mlcommons/inference_results_v6.0
- NVIDIA 曾提交 Hopper 的 MaxQ（功耗）結果：v4.0 的 `DGX-H100_H100-SXM-80GBx8_TRT_MaxQ`，
  v4.1 的 `H200-SXM-141GBx8_TRT_MaxQ`（包含 bert-99.9、dlrm-v2-99.9、gptj-99.9、llama2-70b-99.9、SDXL 等）。
  https://github.com/mlcommons/inference_results_v4.1/tree/main/closed/NVIDIA/results/H200-SXM-141GBx8_TRT_MaxQ 、
  https://github.com/mlcommons/inference_results_v4.0/tree/main/closed/NVIDIA/results/DGX-H100_H100-SXM-80GBx8_TRT_MaxQ
- MLPerf Power 的方法論論文：Tschand 等人，收集 60 個系統的 1,841 筆可重現量測。https://arxiv.org/abs/2410.12032
- **（推論）** 因為量的是整機 AC 功率，GPU 尾數歸零省下的功率會被 CPU、風扇與電源轉換損失稀釋。
  若要進入 Power 類別，應以 Offline 情境的 samples/s/W 呈現。

---

## Q3. FP8 PTQ 準確度文獻：哪些任務與模型有明顯損失

- **Kuzmin 等人**（Qualcomm，NeurIPS 2022）：
  - PTQ 時 FP8 在準確度上普遍優於 INT8，而「the choice of the number of exponent bits is driven by the severity of outliers」。
    QAT 會讓格式間的差異消失。
  - 有大離群值的網路（ViT、BERT、SalsaNext、HRNet）偏好更多指數位元；CNN 偏好更多尾數位元。
  - BERT-base 的 INT8 PTQ 只剩 71.03（FP32 為 83.06），FP8 最佳可達 83.08。
  - https://arxiv.org/abs/2208.09225
- **van Baalen 等人**（Qualcomm, 2023）：在專用硬體上 FP 格式的運算效率比 INT 低約 50–180%，
  並主張 FP8 在推論上不足以取代 INT8。https://arxiv.org/abs/2303.17951
- **Shen 等人**（Intel, 2023）：
  - 涵蓋 75 種網路架構，以「相對 FP32 損失 < 1%」為通過標準。
  - 通過率：E5M2 74.89%、E4M3 static 92.64%（NLP 96.32%）、E3M4 90.04%、INT8 65.87%。
  - E4M3 適合 NLP；E3M4（尾數較多）在 CV 略好。
  - Bloom-7B1 Lambada：FP32 0.5764、E5M2 0.5424（−5.9%）、E4M3 0.5748。
  - CNN 若連第一與最後的算子也量化，E5M2 與 E4M3 的通過率分別掉 25% 與 15%。
  - https://arxiv.org/abs/2309.14592
- **Micikevicius 等人**（NVIDIA/Arm/Intel, 2022）：
  - E4M3 PTQ 在 BERT SQuAD 上保持準確度（BERT-Base F1 88.19→88.09），INT8 則掉到 76.89。
  - GPT3 126M 的 perplexity 19.01→19.43。
  - 不做 scaling 直接 cast 到 FP8，perplexity 會升到 11.0。
  - https://arxiv.org/abs/2209.05433
- **Kurtic 等人**（ACL 2025）：
  - 在 Llama-3.1 全系列上做了超過 50 萬次評估，結論是 W8A8-FP「effectively lossless」，8-bit 平均恢復 99.75%。
  - 他們的 FP8 設定：所有 transformer block 內的 linear 用 per-output-channel 權重與 dynamic per-token activation。
  - 部署建議：同步情境 W4A16 最划算，非同步 continuous batching 則 W8A8 最好。
  - https://arxiv.org/abs/2411.02355
- **Lee 等人**（IJCAI 2025）：FP8 在 1B–405B、各類任務上都是最穩定的方法；但量化模型普遍在 instruction-following 與 hallucination detection 上較弱。
  https://arxiv.org/abs/2409.11055
- **Mekala 等人**（EMNLP 2025，輸入超過 64K 的長 context）：
  - FP8 與 GPTQ-int8 平均只掉 0.2–0.8%（論文兩處數字順序不一致），統計上與 BF16 無顯著差異。
  - 但在 CS4 受限生成任務上，FP8 下降 −1.8%，統計上顯著（p<0.05）。
  - 4-bit 在長輸入上最多掉 59%，非英語輸入的退化更嚴重。
  - https://arxiv.org/abs/2505.20276
- **Liu 等人**（COLM 2025，推理模型）：W8A8（他們用 INT8 的 SmoothQuant，不是 FP8）或 W4A16 可以做到無損，
  更低位元則風險明顯。模型大小、模型來源與任務難度是關鍵因素。https://arxiv.org/abs/2504.04823
- **Llama 3 405B**：標準 benchmark 看起來 FP8 與 BF16 相當，但沒有緩解措施時會偶爾產生損壞的回應，只有看 reward 分布才看得出來。
  https://arxiv.org/abs/2407.21783
- **ML.ENERGY**（Chung 等人, 2026）：在 H100 上 batch size 8–16 時，FP8 反而比 BF16 多用最多 56% 能量、慢最多 26%，
  要到較大 batch 才開始占優，原因是未融合的量化／反量化開銷與 GPU 利用率不足。
  這是效能面的限制，不是準確度問題。https://arxiv.org/abs/2601.22076
- **歸納**：FP8 E4M3 在大型 LLM 的標準 benchmark 上幾乎無損。明顯損失集中在幾類情況：
  - 小模型
  - 沒有 per-token 或 row-wise scale 時的離群 token
  - attention 用 FP8（尤其影片 diffusion 與長 context 檢索）
  - 受限生成等少數任務
  - 尾數只剩 2 位的 E5M2

  對本研究而言，最關鍵的是最後一點：尾數從 3 位減到 2 位，已知會讓 W+A PTQ 的通過率明顯下降。

---

## Q4. 降低尾數位元（而非指數位元）的前人研究

- **尾數與指數的取捨**：
  - Kuzmin 等人：CNN 需要更多尾數位元，有離群值的 transformer 需要更多指數位元。https://arxiv.org/abs/2208.09225
  - MX 論文：在 MXFP8 與 MXFP6 中，「the variant of the format with more mantissa bits was better for direct-cast inference」。
    LLaMA-7B direct-cast 的 wikitext perplexity：FP32 9.488、MXFP8 9.768、MXFP6 9.628、MXFP4 27.201。
    依 Table 5 的說明，這裡的 MXFP8／MXFP6 採用尾數較多的 E4M3／E2M3 變體。
    MX 格式為 32 個元素共用一個 E8M0 scale；MXFP6 有 E2M3 與 E3M2 兩種，MXFP4 為 E2M1。https://arxiv.org/abs/2310.10537
  - DeepSeek-V3 的「Mantissa over Exponents」：有了細粒度 scale 之後，所有 tensor 都用 E4M3 而不用 E5M2。https://arxiv.org/abs/2412.19437
  - **含意**：有細粒度 scale 時，尾數位元比指數位元更值錢。這不利於 E4M3 保留 2／1／0 個尾數位元的方案，
    因為指數位元沒減、尾數卻減了。反過來，BF16 有 8 個指數位元，把尾數減到 2–3 位，形同「不需要 scale 的 E8M3」，
    這個方向比較合理（推論）。
- **FP6-LLM／Quant-LLM**（USENIX ATC 2024）：
  - FP6 權重量化「preserve the model quality consistently across varied applications」，比 4-bit 好，不需要細粒度量化。
    LLaMA-70B 可在單張 GPU 上跑，推論吞吐量比 FP16 高 1.69–2.65 倍。
  - 反量化方式是把 FPx 的尾數「padding zeros」轉成 FP16（Eq. 2），再在送進 tensor core 前乘上 per-channel scale。
  - **（推論）** 如果改把 scale 移到 GEMM 輸出端，FP16 權重運算元就會帶著大量為零的低位尾數，正好是本研究的功耗機制；
    論文本身沒有討論功耗。
  - https://arxiv.org/abs/2401.14112 、https://www.usenix.org/conference/atc24/presentation/xia
- **NVFP4**（NVIDIA 技術部落格, 2025）：
  - 格式為 E2M1，每 16 個元素共用一個 E4M3 scale，每個 tensor 再有一個 FP32 scale。
  - DeepSeek-R1-0528 從 FP8 換成 NVFP4，準確度下降不超過 1%。MLPerf 的 Llama2-70B-99.9 在 Blackwell 上也用 fp4 通過（見 2.3）。
  - 這說明只要 scale 夠細，1 個尾數位元也夠用。但 Hopper 沒有原生的 block-scaled FP8 GEMM。
  - https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/
- **ZeroQuant-FP**（Microsoft, 2023）：FP8 activation 一致優於 INT8；FP4 權重與 INT4 相當甚至更好。https://arxiv.org/abs/2307.09782
- **Anda**（HPCA 2025）：對 FP activation 採用群組共用指數與動態尾數位元分配，逐模組搜尋位寬。
  搭配 bit-serial 硬體，平均加速 2.4 倍、面積效率 4.0 倍、能源效率 3.1 倍。
  這是最接近「推論時依模組調整尾數長度」的前例，但它針對的是 ASIC。https://arxiv.org/abs/2411.15982
- **BitMoD**（HPCA 2025）：每 128 個權重一組，各自選用不同資料型別，搭配 bit-serial PE。
  權重 4-bit 時判別任務平均損失 < 0.5%，生成任務可做到 3-bit。https://doi.org/10.1109/HPCA61900.2025.00084
  （摘要取自 Semantic Scholar API）
- **位元層級稀疏性加速器**：
  - Bit-Pragmatic（MICRO 2017）：CNN 的 activation 平均有 92% 的位元為零，用 shift-and-add 跳過這些零位元（定點數）。
    https://doi.org/10.1145/3123939.3123982
  - FPRaker（MICRO 2021）：把浮點 significand 拆成 signed powers of two，跳過無效的項，針對訓練。
    https://arxiv.org/abs/2010.08065 、https://doi.org/10.1145/3466752.3480106
  - AdaptivFloat（DAC 2020）：逐層調整的浮點編碼。https://doi.org/10.1109/DAC18072.2020.9218516
  - 以上都是用專用硬體「跳過」零位元來換速度，不是在現成 GPU 上靠降低翻轉來換時脈。
- **利用零值或低翻轉降低功耗**：Peltekis 等人（MOCAST）在 BF16 systolic array 的資料輸入路徑上結合 bus-invert coding 與 zero-value clock gating，
  CNN 的整體動態功耗降低 6.2–9.4%。https://arxiv.org/abs/2304.12691
- **在現成 GPU 上用尾數歸零換取推論時脈**：除了 Gregersen 2024 的觀察之外，沒找到直接前例。
  搜尋關鍵字包含 mantissa truncation、bit toggling、tensor core、inference、power cap 等組合，截至 2026-09-29。**未查證是否真的不存在**。

---

## Q5. 資料相依與位元翻轉功耗（Gregersen 2024、Bhalachandra 2022 之外）

- **ALUPower**（Lucas & Juurlink, MASCOTS 2016）：在 GTX 580 與 GTX 750 Ti 上，同一個 kernel 依處理的數值不同，功耗介於 155–257 W，
  並以 Hamming distance 建立 ALU 能耗模型。https://www.static.tu.berlin/fileadmin/www/10002217/publications-pdf/alupower.pdf 、
  https://doi.org/10.1109/MASCOTS.2016.21
- **MEMPower**（Lucas & Juurlink, ARCS 2019）：資料相依的 GPU 記憶體存取能耗模型。
  **（推論）** 這與 decode 階段（memory-bound）有關：權重尾數歸零可能降低 HBM 與互連的翻轉能耗。
  https://doi.org/10.1007/978-3-030-18656-2_15 （摘要取自 Semantic Scholar API）
- **Horace He 的部落格**（2024-04-29，非同儕審查）：
  - A100（該卡功耗上限 330 W）上，全零輸入約 295 TFLOPS、randn 約 257 TFLOPS，差距約 15%，也測了 ones、稀疏、單一位元、三值等輸入。
  - 功耗上限從 330 W 往下調到 100 W 的過程中，「可預測輸入」的相對優勢跟著變大，但在最低的 100 W 反而逆轉。
    這與本專案「越接近功耗上限，尾數歸零效益越大」的觀察一致。
  - 他的解釋是電晶體切換造成的動態功耗，並說「Matmuls on the H100 are primarily not compute or bandwidth limited, they are power limited」。
  - 這是與本研究機制（資料→功耗→時脈）最接近的公開觀察，但沒有討論尾數歸零與準確度。
  - https://www.thonking.ai/p/strangely-matrix-multiplications
- **POLCA**（Patel 等人, ASPLOS 2024）：
  - Insight 4：prompt 階段短但「typically reach or exceed GPU TDP」，token 階段較長且功耗較低。
  - Insight 7：power capping 是被動反應，主要只影響 prompt 階段。
  - https://www.microsoft.com/en-us/research/wp-content/uploads/2024/03/GPU_Power_ASPLOS_24.pdf 、https://doi.org/10.1145/3620666.3651329
- **Splitwise**（ISCA 2024）：prompt 階段是 compute-intensive，token 階段是 memory-intensive，兩者的功耗特性不同；
  把兩個階段拆到不同機器，在相同功耗與成本下吞吐量可達 2.35 倍。https://arxiv.org/abs/2311.18677
- **Ma 等人**（2026）：在 H200 上，decode 對 GQA、MLA、Gated DeltaNet、Mamba2 都只用 700 W 中的 137–300 W，
  所以 power cap 在 decode 根本不會觸發，他們建議改用 clock locking。https://arxiv.org/abs/2605.11999
- **ML.ENERGY**：FP8 與 BF16 的能耗高低取決於 batch size（見 Q3）。https://arxiv.org/abs/2601.22076
- **FP8 tensor core 的資料相依功耗**：沒找到已發表的量測。Gregersen 2024 量的是 FP32、FP16、FP16-TC、INT8。
  本專案在 GH200 上的 FP8 尾數歸零量測，可能是第一筆公開數據（**未查證是否有未公開或非英文的前例**）。

---

## Q6. Hopper 上 FP8 以外的其他加速手段（簡述）

- **峰值算力**：H100 SXM 的 BF16/FP16 為 1,979 TFLOPS，FP8 為 3,958 TFLOPS，INT8 為 3,958 TOPS，
  但標示「With sparsity」，也就是 dense 各為一半。所以在 Hopper 上 INT8 與 FP8 的算力相同。https://www.nvidia.com/en-us/data-center/h100/
- **2:4 structured sparsity**：
  - SparseGPT 在 OPT-175B／BLOOM-176B 上做到 60% 非結構化稀疏，perplexity 幾乎不增加，也能套用 2:4 與 4:8。https://arxiv.org/abs/2301.00774
  - 沒有重新訓練時，2:4 的代價很大：LLaMA-7B 的 wikitext perplexity 從 5.68 升到 11.00（SparseGPT）或 11.53（Wanda）；
    LLaMA-2-70B 從 3.12 升到 5.40／5.16；LLaMA-2-70B 的 zero-shot 平均從 67.08 降到 63.89／64.14。
  - 在 A6000 上，LLaMA-65B 的 linear 層約加速 1.6 倍，LLaMA-7B 端到端加速 1.24 倍。https://arxiv.org/abs/2306.11695
  - Red Hat／Neural Magic 的 2:4 Sparse Llama 3.1 8B 經過 13B tokens 的蒸餾再訓練，OpenLLM v1 恢復 98.37%。
    搭配 FP8，在 H100 上相對 BF16 延遲低 1.7 倍、吞吐量高 1.5 倍；GEMM 層級相對 dense FP8 延遲最多降低 30%。
    https://developers.redhat.com/articles/2024/12/18/24-sparse-llama-fp8-sota-performance-nvidia-hopper-gpus 、
    https://huggingface.co/RedHatAI/Sparse-Llama-3.1-8B-2of4
  - MLPerf closed division 禁止剪枝（見 2.5）。
- **INT8 SmoothQuant**：把 activation 的離群值平移到權重，W8A8 準確度損失可忽略，最多 1.56 倍加速、2 倍記憶體節省。
  https://arxiv.org/abs/2211.10438 。在 Hopper 上它相對 FP8 沒有算力優勢；調校得好的 W8A8-INT 相對 BF16 仍損失約 1–3%，
  FP8 則幾乎無損。
  https://arxiv.org/abs/2411.02355
- **INT4 weight-only（AWQ、GPTQ）**：
  - AWQ（MLSys 2024）：只保護 1% 的重要權重，TinyChat 比 HF FP16 快 3 倍以上。https://arxiv.org/abs/2306.00978
  - GPTQ（ICLR 2023）：3–4 bit，在 A100 上端到端加速約 3.25 倍、A6000 上 4.5 倍。https://arxiv.org/abs/2210.17323
  - 這類加速來自 memory-bound decode 減少了權重讀取，而不是算力提升；Kurtic 等人也建議同步小批次情境用 W4A16。
    https://arxiv.org/abs/2411.02355
  - **（推論）** 這與尾數歸零的作用區間互補：尾數歸零只在 compute-bound 且觸發功耗上限的 GEMM 上有效。
- **量化 speculative decoding**：QSpec（EMNLP 2025）以 W4A4 起草、W4A16 驗證，最多加速 1.64 倍且品質不降。https://aclanthology.org/2025.emnlp-main.240/

---

## 可引用文獻（僅列已查證者）

資料相依功耗與 LLM 功耗特性
- Jan Lucas, Ben Juurlink. *ALUPower: Data Dependent Power Consumption in GPUs.* MASCOTS 2016. doi:10.1109/MASCOTS.2016.21.
  https://www.static.tu.berlin/fileadmin/www/10002217/publications-pdf/alupower.pdf
- Jan Lucas, Ben Juurlink. *MEMPower: Data-Aware GPU Memory Power Model.* ARCS 2019, LNCS. doi:10.1007/978-3-030-18656-2_15.
- Horace He. *Strangely, Matrix Multiplications on GPUs Run Faster When Given "Predictable" Data!* 部落格, 2024-04-29.
  https://www.thonking.ai/p/strangely-matrix-multiplications （非同儕審查）
- Pratyush Patel, Esha Choukse, Chaojie Zhang, Íñigo Goiri, Brijesh Warrier, Nithish Mahalingam, Ricardo Bianchini.
  *Characterizing Power Management Opportunities for LLMs in the Cloud.* ASPLOS 2024. doi:10.1145/3620666.3651329.
- Pratyush Patel, Esha Choukse, Chaojie Zhang, Aashaka Shah, Íñigo Goiri, Saeed Maleki, Ricardo Bianchini.
  *Splitwise: Efficient Generative LLM Inference Using Phase Splitting.* ISCA 2024. https://arxiv.org/abs/2311.18677
- Bole Ma, Ayesha Afzal, Jan Eitzinger, Gerhard Wellein. *The Illusion of Power Capping in LLM Decode: A Phase-Aware Energy
  Characterisation Across Attention Architectures.* arXiv:2605.11999 (2026). https://arxiv.org/abs/2605.11999
- Jae-Won Chung, Ruofan Wu, Jeff J. Ma, Mosharaf Chowdhury. *Where Do the Joules Go? Diagnosing Inference Energy Consumption.*
  arXiv:2601.22076 (2026). https://arxiv.org/abs/2601.22076
- C. Peltekis, D. Filippas, G. Dimitrakopoulos, C. Nicopoulos. *Low-Power Data Streaming in Systolic Arrays with Bus-Invert Coding and
  Zero-Value Clock Gating.* MOCAST; arXiv:2304.12691. https://arxiv.org/abs/2304.12691
- Arya Tschand, Arun Tejusve Raghunath Rajan, Sachin Idgunji, … Vijay Janapa Reddi. *MLPerf Power: Benchmarking the Energy Efficiency
  of Machine Learning Systems from Microwatts to Megawatts for Sustainable AI.* arXiv:2410.12032. https://arxiv.org/abs/2410.12032

FP8 格式與 PTQ 準確度
- Paulius Micikevicius, Dusan Stosic, Neil Burgess, Marius Cornea, Pradeep Dubey, … Hao Wu. *FP8 Formats for Deep Learning.*
  arXiv:2209.05433 (2022). https://arxiv.org/abs/2209.05433
- Andrey Kuzmin, Mart van Baalen, Yuwei Ren, Markus Nagel, Jorn Peters, Tijmen Blankevoort. *FP8 Quantization: The Power of the
  Exponent.* NeurIPS 2022. https://arxiv.org/abs/2208.09225
- Mart van Baalen, Andrey Kuzmin, Suparna S Nair, Yuwei Ren, Eric Mahurin, Chirag Patel, … Tijmen Blankevoort.
  *FP8 versus INT8 for efficient deep learning inference.* arXiv:2303.17951 (2023). https://arxiv.org/abs/2303.17951
- Haihao Shen, Naveen Mellempudi, Xin He, Qun Gao, Chang Wang, Mengni Wang. *Efficient Post-training Quantization with FP8
  Formats.* arXiv:2309.14592 (2023). https://arxiv.org/abs/2309.14592
- Xiaoxia Wu, Zhewei Yao, Yuxiong He. *ZeroQuant-FP: A Leap Forward in LLMs Post-Training W4A8 Quantization Using Floating-Point
  Formats.* arXiv:2307.09782 (2023). https://arxiv.org/abs/2307.09782
- Eldar Kurtic, Alexandre Marques, Shubhra Pandit, Mark Kurtz, Dan Alistarh. *"Give Me BF16 or Give Me Death"? Accuracy-Performance
  Trade-Offs in LLM Quantization.* ACL 2025. https://arxiv.org/abs/2411.02355
- Jemin Lee, Sihyeong Park, Jinse Kwon, Jihun Oh, Yongin Kwon. *Exploring the Trade-Offs: Quantization Methods, Task Difficulty, and
  Model Size in Large Language Models From Edge to Giant.* IJCAI 2025. https://arxiv.org/abs/2409.11055
- Anmol Mekala, Anirudh Atmakuru, Yixiao Song, Marzena Karpinska, Mohit Iyyer. *Does quantization affect models' performance on
  long-context tasks?* EMNLP 2025. https://arxiv.org/abs/2505.20276
- Ruikang Liu, Yuxuan Sun, Manyi Zhang, Haoli Bai, Xianzhi Yu, Tiezheng Yu, … Lu Hou. *Quantization Hurts Reasoning? An Empirical
  Study on Quantized Reasoning Models.* COLM 2025. https://arxiv.org/abs/2504.04823
- Tanishq Kumar, Zachary Ankner, Benjamin F. Spector, Blake Bordelon, Niklas Muennighoff, … Aditi Raghunathan. *Scaling Laws for
  Precision.* arXiv:2411.04330 (2024). https://arxiv.org/abs/2411.04330
- Llama Team, AI @ Meta (Aaron Grattafiori, Abhimanyu Dubey, …). *The Llama 3 Herd of Models.* arXiv:2407.21783 (2024).
  https://arxiv.org/abs/2407.21783
- DeepSeek-AI. *DeepSeek-V3 Technical Report.* arXiv:2412.19437 (2024). https://arxiv.org/abs/2412.19437
- Maxim Fishman, Brian Chmiel, Ron Banner, Daniel Soudry. *Scaling FP8 training to trillion-token LLMs.* arXiv:2409.12517 (2024).
  https://arxiv.org/abs/2409.12517
- Penghui Qi, Zichen Liu, Xiangxin Zhou, Tianyu Pang, Chao Du, Wee Sun Lee, Min Lin. *Defeating the Training-Inference Mismatch via
  FP16.* arXiv:2510.26788 (2025). https://arxiv.org/abs/2510.26788

離群值與 attention
- Tim Dettmers, Mike Lewis, Younes Belkada, Luke Zettlemoyer. *LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale.*
  NeurIPS 2022. https://arxiv.org/abs/2208.07339
- Mingjie Sun, Xinlei Chen, J. Zico Kolter, Zhuang Liu. *Massive Activations in Large Language Models.* COLM 2024.
  https://arxiv.org/abs/2402.17762
- Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao. *FlashAttention-3: Fast and Accurate Attention with
  Asynchrony and Low-precision.* NeurIPS 2024（venue 取自 Semantic Scholar）. https://arxiv.org/abs/2407.08608
- Jintao Zhang, Jia Wei, Haofeng Huang, Pengle Zhang, Jun Zhu, Jianfei Chen. *SageAttention: Accurate 8-Bit Attention for
  Plug-and-play Inference Acceleration.* ICLR 2025. https://arxiv.org/abs/2410.02367
- Jintao Zhang, Haofeng Huang, Pengle Zhang, Jia Wei, Jun Zhu, Jianfei Chen. *SageAttention2: Efficient Attention with Thorough Outlier
  Smoothing and Per-thread INT4 Quantization.* ICML 2025. https://arxiv.org/abs/2411.10958

低尾數格式與位元層級硬體
- Haojun Xia, Zhen Zheng, Xiaoxia Wu, Shiyang Chen, Zhewei Yao, Stephen Youn, … Shuaiwen Leon Song. *Quant-LLM: Accelerating the
  Serving of Large Language Models via FP6-Centric Algorithm-System Co-Design on Modern GPUs.* USENIX ATC 2024
  （arXiv 版標題為 FP6-LLM）. https://www.usenix.org/conference/atc24/presentation/xia 、https://arxiv.org/abs/2401.14112
- Bita Darvish Rouhani, Ritchie Zhao, Ankit More, Mathew Hall, Alireza Khodamoradi, Summer Deng, … Eric Chung. *Microscaling Data
  Formats for Deep Learning.* arXiv:2310.10537 (2023). https://arxiv.org/abs/2310.10537
- Eduardo Alvarez et al. *Introducing NVFP4 for Efficient and Accurate Low-Precision Inference.* NVIDIA Technical Blog, 2025-06-24.
  https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/
- Chao Fang, Man Shi, Robin Geens, Arne Symons, Zhongfeng Wang, Marian Verhelst. *Anda: Unlocking Efficient LLM Inference with a
  Variable-Length Grouped Activation Data Format.* HPCA 2025. https://arxiv.org/abs/2411.15982
- Yuzong Chen, Ahmed F. AbouElhamayed, Xilai Dai, Yang Wang, Marta Andronic, George A. Constantinides, Mohamed S. Abdelfattah.
  *BitMoD: Bit-serial Mixture-of-Datatype LLM Acceleration.* HPCA 2025. doi:10.1109/HPCA61900.2025.00084.
- Jorge Albericio, Alberto Delmás, Patrick Judd, Sayeh Sharify, Gerard O'Leary, Roman Genov, Andreas Moshovos.
  *Bit-Pragmatic Deep Neural Network Computing.* MICRO 2017. doi:10.1145/3123939.3123982.
- Omar Mohamed Awad, Mostafa Mahmoud, Isak Edo, Ali Hadi Zadeh, Ciaran Bannon, Anand Jayarajan, Gennady Pekhimenko, Andreas Moshovos.
  *FPRaker: A Processing Element For Accelerating Neural Network Training.* MICRO 2021. doi:10.1145/3466752.3480106.
- Thierry Tambe, En-Yu Yang, Zishen Wan, Yuntian Deng, Vijay Janapa Reddi, Alexander M. Rush, David Brooks, Gu-Yeon Wei.
  *Algorithm-Hardware Co-Design of Adaptive Floating-Point Encodings for Resilient Deep Learning Inference.* DAC 2020.
  doi:10.1109/DAC18072.2020.9218516.

其他加速手段
- Elias Frantar, Dan Alistarh. *SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot.* arXiv:2301.00774 (2023).
  https://arxiv.org/abs/2301.00774
- Mingjie Sun, Zhuang Liu, Anna Bair, J. Zico Kolter. *A Simple and Effective Pruning Approach for Large Language Models* (Wanda).
  ICLR 2024. https://arxiv.org/abs/2306.11695
- Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, Song Han. *SmoothQuant: Accurate and Efficient Post-Training
  Quantization for Large Language Models.* ICML 2023. https://arxiv.org/abs/2211.10438
- Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, … Song Han. *AWQ: Activation-aware Weight Quantization for LLM
  Compression and Acceleration.* MLSys 2024. https://arxiv.org/abs/2306.00978
- Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh. *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained
  Transformers.* ICLR 2023. https://arxiv.org/abs/2210.17323
- Juntao Zhao, Wenhao Lu, Sheng Wang, Lingpeng Kong, Chuan Wu. *QSpec: Speculative Decoding with Complementary Quantization Schemes.*
  EMNLP 2025. https://aclanthology.org/2025.emnlp-main.240/

規則、文件與結果資料（非論文）
- MLCommons. *MLPerf Inference Rules* 與 *Power Measurement Rules*（inference_policies, master，2026-09-29 抓取）.
  https://github.com/mlcommons/inference_policies
- MLCommons. *inference_results_v4.0 / v4.1 / v5.0 / v5.1 / v6.0*（NVIDIA 的 configs、code README、accuracy.txt、summary_results.json）.
  https://github.com/mlcommons
- NVIDIA Model Optimizer（repo 已從 TensorRT-Model-Optimizer 改名）. https://github.com/NVIDIA/Model-Optimizer
- NVIDIA Transformer Engine 文件（1.13 版 FP8 primer、2.4 版 attention）；TensorRT-LLM 文件；vLLM llm-compressor FP8 文件（URL 見正文）。

已在 `references.md` 的 Gregersen 2024 與 Bhalachandra 2022 不重列。

---

## 未查證或待補

- 科學 ML 代理模型（天氣、PDE 等）的 FP8 推論精度：沒找到可開啟的一手來源。
- MLPerf Whisper 在 GPU 上一律用 FP16/BF16 的原因：README 沒有說明。
- OCP Microscaling Formats v1.0 規格書：官方頁面被 Cloudflare 擋住，無法開啟。格式定義改引 MX 論文（arXiv:2310.10537）的表格。
- 以下論文的正式發表 venue 未確認，一律以 arXiv 引用：Shen 2023、Micikevicius 2022、SparseGPT、Fishman 2024、Kumar 2024。
- LLM.int8() 中離群特徵出現的模型規模門檻（常被引用為 6.7B）：本次只開了摘要，沒有核對全文，所以正文不引用這個數字。
- MLPerf v6.0 的 Power 結果：summary_results.json 只有 520 筆，也沒有 has_power 條目，檔案可能不完整。
- 「只對單一運算元（權重）做尾數歸零」能省下多少 tensor core 功耗：沒有文獻，需要本專案自行量測。
- 有細粒度 scale 的 E4M2／E4M1（FP8 容器）在 LLM 上的 W+A 準確度：沒找到直接評估。
  最接近的是 E5M2（Shen）、MXFP6 E3M2／E2M3（MX 論文）、NVFP4 E2M1，都不是同一種格式。

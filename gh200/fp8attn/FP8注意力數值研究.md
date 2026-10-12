# FP8 attention 數值一致性研究（SCC26 MLPerf Llama2-70B LoRA）

2026-10-06。研究成果，**不在提交設定內**；提交仍用 NVIDIA v5.0 的 TE/cuDNN FP8 attention（下稱 A）。
細節與 job id：`實驗日誌.md`。

## 一句話

我們在真實的 70B LoRA 訓練中量到 cuDNN FP8 attention backward 的 forward／backward 不一致（論文所稱 stale delta），
證明它在原生 FP8 下是無方向雜訊、不傷收斂；自己實作了論文的修正（Delta-Matching），在 kernel 與訓練中驗證；
並發現截尾數會把這個雜訊變成同號偏差，而 Delta-Matching 能完全消除它。過程中另外找到 FP8 delayed scaling
在訓練開頭讓梯度全部下溢、以及穩態常飽和的問題，並給出修正。

## 背景

- 論文：Delta-Matching（arXiv 2609.37852）。FlashAttention backward 用 δᵢ = ⟨dOᵢ, Oᵢ⟩；FP8 下 forward 與 backward
  各自量化，dS = P∘(dP − δ) 的列和不再為 0，偏差累積會讓 pretraining 退化。修正：δᵢ = Σⱼ Pᵢⱼ·dP̂ᵢⱼ，列和精確為 0。
- 我們的條件：GH200 × 14、每節點 1 GPU、功耗上限 575 W、LoRA 約 280 步，收斂點已在 BF16 的 RCP 平均（3842 vs 3819）。

## 量測與結果

### 1. 我們的 FP8 attention 有 stale delta，但無方向

在訓練中擷取 4 層 × 多個 step 的 q/k/v/dO 與 TE 實際輸出的梯度，依 TE 2.2 recipe 逐步重現：

- TE 輸出的 dq 與 stale 版重現只差 1–2.4%（與 Delta-Matching 版差 5.5–8.9%），確認 cuDNN 用 stale δ。
- 列和殘差 1.4–2.4e-2（論文 1e-2–5e-2 的低端）；有號平均只有絕對值的 ±2%，正負各半。
- → 零均值雜訊，在 8192×GBS 個 token 上平均掉；解釋了為何我們的 FP8 能追平 BF16 的 RCP。

### 2. 自寫 FP8 attention + Delta-Matching（Triton，Hopper）

- kernel 測試（`test_minimal.py`，隨機與 3 組真實張量）：列和殘差 2–4e-2 → **約 1e-7**；dQ 誤差降 10–19%。
- 接進 70B 訓練：可跑、loss 與 A 同範圍、完全可重現。
- 速度：575 W 下比 A 慢約 7%（attention 每步 0.97 s vs 0.73 s）。我們的 Triton kernel 在約 1300 MHz 下
  每個 kernel 慢約 40%，cuDNN 沒有；在功耗受限下 kernel 效率才是瓶頸。另外 Triton 3.8 與容器的
  torch 2.7 `torch.compile` 不相容，測試時兩邊都關掉編譯。

### 3. 截尾數讓 stale delta 變成同號偏差，Delta-Matching 消除它

把 attention 的 FP8 運算元尾數從 3 位截到 k 位（`test_mant_trunc.py`）：

| 尾數位元 | stale 系統偏差 | Delta-Matching 偏差 |
|---|---|---|
| 3（原生） | 正負都有，≈ 0 | ≈ 1e-8 |
| 2 | −4.3 ~ −4.9e-2 | ≈ 1e-8 |
| 1 | −1.2 ~ −1.3e-1 | ≈ 1e-8 |
| 0 | −2.5 ~ −2.7e-1 | ≈ 1e-8 |

- 這是論文描述的有害型態，由截斷產生；Delta-Matching 在任何位元數都能消除，是「attention 截尾數」的必要條件。
- 但在這個工作負載上截 attention 運算元本身不划算：訓練中時脈只高約 20 MHz，功耗大頭在凍結權重的 GEMM。

### 4. FP8 delayed scaling 的冷啟動與飽和

- warmup 後 NVIDIA 程式碼把所有 FP8 scale 重設為 1：step 1 每層進入 attention 的 dO 全為 0，step 2 dq、dk 全為 0，
  step 3 飽和 10–30 倍（誤差 19–94%）。這比 stale delta 大一個量級。
- 穩態也常飽和：dO／dP／dQKV 在 18／19／14% 的（層，步）超出可表示範圍。
- 修正（工作副本，預設關閉）：`FP8_MARGIN=2`（原程式沒有把 margin 傳下去）＋計時內的校準 pass（梯度丟棄、不更新權重）。
  結果：step 1–3 不再全零或飽和，穩態飽和降到 0.3–0.9%；每步沒有成本，一次性校準約佔 TTT 的 0.35–0.65%。
- 合規：margin 屬精度設定（「Any type of precision is okay」）；校準 pass 我們判斷合法，但規則沒有直接寫到。

## 對提交的結論

- 提交維持 A。Delta-Matching 在我們的條件下不加速（慢 7%），收斂面又被 RCP 封頂（最多約 0.1 分鐘）。
- 冷啟動修正改善正確性，不改善分數；採用前需要多 seed 收斂與 RCP 驗證。
- 若要讓 FP8 + Delta-Matching 變快，需要 CUDA/CUTLASS 重寫 backward，讓 exp 與 tensor core 重疊（估計熟手 2–4 週）。

## 檔案

- `gh200/fp8attn/`：`fp8attn2.py`（kernel）、`test_minimal.py`、`test_mant_trunc.py`、`residual_measure.py`、
  `fp8_scale_fix.py`、`slurm/`、`實驗日誌.md`
- 訓練整合（工作副本，非提交程式碼）：`/nfs/home/koying/mlperf-fp8attn-work/nvidia_impl/`（`fp8dm_attn.py`，開關
  `ATTN_IMPL=fp8dm`、`FP8DM_MATCHED`、`FP8DM_MANT_BITS`、`FP8_MARGIN`、`FP8_BOOTSTRAP_PASSES`）
- 真實張量 dump：`/nfs/home/koying/fp8attn-dumps/`（14 GB）

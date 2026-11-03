# 交接給 cold-gemm：MI350X 上的 AI（LLM）實驗結果

2026-11-03。來源：mantissa-optimize repo、分支 `ai-fp8-paper`（worktree `.claude/worktrees/ai-fp8-paper`），所有路徑相對 `mi350/`。
使用者決定：「合進 cold-gemm 那篇」——這邊的 AI 結果當作 cold-gemm 論文的 LLM 端到端與數值章節，不另寫一篇。
過程細節（job id、每一步的決定）在 `ai/實驗日誌.md`；白話進度頁 https://claude.ai/artifact/VffxbWrAEeyapRPYL9DQw8。

## 一句話結論

尾數截斷（BF16、FP8 低位清零）在 MI350X 的 AI 工作負載上只換到 2–6% 的速度或能量，而且只對 BF16 有效；FP8 本身就比 BF16
少約 29% 能量。精度結論跨平台（GH200／MI350X）一致。FP8 attention 的運算元截斷會造成有方向的梯度偏差（GH200 發現，MI350X 重現），
四捨五入可縮到 1/4–1/5、Delta-Matching 可消除。→ 正負號編碼（cold-gemm 的主軸）是比尾數截斷更有前景的方向。

## 結果與檔案

| 項目 | 結果 | 檔案／程式 |
|---|---|---|
| 精度：WikiText-2 perplexity，7 模型 × 19 格式 | BF16 留 4 位 +0.09–0.45%、3 位 +0.34–1.94%、3 位朝零截斷 +5.9–11.6%、FP8 per-tensor −0.16–2.3%、FP8 只截權重到 2 位 +1.3–4.0%；與 GH200 相差多 < 0.1 個百分點 | `results/ai_ppl.csv`、`ai/ppl_summary.py`、`scripts/ai_ppl.sh`、`../gh200/infer/ppl_eval.py`（新增 `--wikitext-parquet`） |
| GEMM：只截權重 vs 兩邊都截（Llama-3-8B MLP 形狀，8192×14336×4096 與轉置） | 1000 W：BF16 3 位 +3.7–4.0%（只權重）／+7.5–8.0%（兩邊），FP8 2 位 +1.2–1.5%／+3.0–3.5%；700 W 約 1.5 倍 | `results/ai_gemm_wo.csv`、`ai/gemm_wo_summary.py`、`scripts/ai_gemm_wo.sh` |
| decode 能量（Llama 2 7B 224 個線性層的真實權重依序串流，batch 1–256，1000／600 W，整台同格式 ABBA） | 1000 W 每 token 能量：BF16 3 位 −2.3%（b1）～−6.3%（b256），FP8 2 位 ≈ 0～−1%；FP8 原生對 BF16 原生 −29%（b1）、b256 快 43% | `results/ai_decode.csv`、`ai/decode_bench.py`、`ai/decode_summary.py`、`scripts/ai_decode.sh` |
| LoRA 微調（Llama 2 7B、MLPerf GovReport、MLPerf 70B LoRA 設定縮到單卡） | 整台 8 卡同設定 ABBA：權重 3 位每步 +2.5%（2.1–3.1%）；只開 2 卡 +4–5%；收斂未跑（GH200 70B：每步 +1%、樣本 +22–39%） | `results/ai_lora_speed.csv`、`ai/lora.py`、`ai/lora_speed.py`、`scripts/ai_lora*.sh` |
| FP8 attention 數值（Triton FP8 flash attention，fwd＋bwd） | stale delta 的同號偏差，朝零截斷 2／1／0 位：亂數 −0.048／−0.128／−0.256；Llama 2 70B 真實張量（層 0/20/40/79）−0.043～−0.046／−0.123～−0.126／−0.254～−0.264（GH200 −0.043～−0.049／−0.12～−0.13／−0.25～−0.27）。四捨五入：亂數 −0.014／−0.027／−0.057，70B −0.005～−0.010／−0.017～−0.028／−0.036～−0.060（朝零截斷的 1/5～1/10）。原生 3 位 70B ±0.002（零均值）。Delta-Matching 約 1e-7。dQ 誤差四捨五入 2 位就 12–28% | `results/ai_fp8attn_randn.md`、`results/ai_fp8attn_llama70b.md`、`ai/fp8attn_trunc.py`、`ai/attn_dump.py`（70B dump 在 `~/koying/sw/fp8attn-dumps`，job 410）、`../gh200/fp8attn/` |

## 對 cold-gemm 可能直接有用的

1. **decode 能量量測框架**（`ai/decode_bench.py`）：載入真實模型的所有線性權重、依層序串流，量 tokens/s 與每 token 能量。
   cold-gemm 的編碼（例如 w7asym、非負權重＋zero-point）可以直接當成新的 `fmt` 加進 `build()`，在真實權重上量 decode 端到端。
2. **perplexity 量測**（`../gh200/infer/ppl_eval.py`、`scripts/ai_ppl.sh`）：7 個模型（Llama 2 7B／70B、Llama 3.1 8B、Qwen2.5-7B、
   Qwen3-8B、Mistral-7B、OPT-6.7B）已在 `~/koying/sw/hf`；加一個格式就能量編碼的精度代價，30 分鐘跑完 7 個模型 × 19 格式。
3. **FP8 attention 數值**：截斷／編碼若碰到 attention 的 FP8 運算元，要檢查 stale-delta 偏差（`ai/fp8attn_trunc.py` 可直接加變換）。

## 踩過的坑（量測方法）

- **同一節點混跑兩種設定會不對稱地互相干擾**：4 卡原生＋4 卡截斷同時跑時，原生行程每 micro-batch 有約 0.2 s 的 GPU 閒置，
  加速被誇大成 16–22%（實際 2.5%）；鎖頻時也有，只開 2 卡就沒有。速度比較要整台同設定（cold-gemm 的 singleton 規則正好避開）。
- **PyTorch 2.13（ROCm 7.1 wheel）的 HIP graph**：同時存在多個 graph 或 224 個運算的大 graph 在 replay 時 segfault；改用 eager，
  BF16 經 profiler 確認 GPU-bound，FP8 batch ≤ 64 約 25% 是 launch 閒置（活化值量化要預先做或併入前一個 kernel）。
- **eager hipBLASLt 小 batch 讀權重只有約 2.6 TB/s**（HBM 峰值約 1/3）；高效率 decode kernel 下記憶體能量占比會不同，絕對省電比例要謹慎解讀。
- 環境：venv `~/koying/sw/ai-venv`（torch 2.13.0+rocm7.1、transformers 5.22、peft 0.21）；HF 快取 `~/koying/sw/hf`（已登入，Llama 授權已核准）；
  離線時 `datasets` 認不得 hub 快取，直接讀 parquet；bash 的 `SECONDS` 是內建變數。
- 待查：Llama 3.1 8B 原生 perplexity MI350X 6.24 對 GH200 5.61（相對變化一致，應是 transformers 版本差異）。

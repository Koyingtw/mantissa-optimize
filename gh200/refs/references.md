# 參考文獻（已查證）

每一篇都在 2026-09-29 以搜尋結果與原始頁面（arXiv／出版社頁面）核對過標題、作者與出處。
「與本研究的關係」是撰寫論文時的定位依據。

## 1. 資料相依的 GPU 功耗（最直接的前人研究）

- **[Gregersen2024]** Theo Gregersen, Pratyush Patel, Esha Choukse. *Input-Dependent Power Usage in GPUs.*
  SC24 Workshops（SustainSC）; arXiv:2409.18324. https://arxiv.org/abs/2409.18324
  - A100 PCIe（另在 H100、V100、RTX 6000 重現），CUTLASS GEMM 2048×2048（刻意避免降頻），FP32/FP16/FP16-TC/INT8。
  - 變數：數值分佈、位元相似度（隨機化／歸零最低或最高位元）、排列、稀疏度；功耗最多差近 40%。
  - T5「隨機化越多最低位元，功耗越高」、T14「最低位元歸零可降低功耗」。假說：位元翻轉；驗證機制列為未來工作。
  - **關係**：未做功耗受限下的效能、數值精度、真實應用；本研究 M1 以「填 1／固定樣式 vs 填 0」區分翻轉與零值。

- **[Bhalachandra2022]** Sridutt Bhalachandra, Brian Austin, Samuel Williams, Nicholas J. Wright (LBNL).
  *Understanding the Impact of Input Entropy on FPU, CPU, and GPU Power.* arXiv:2212.08805. https://arxiv.org/abs/2212.08805
  - A100（Perlmutter）DGEMM 同尺寸不同數值功耗差 67%（隨機約 400 W vs 固定約 239 W）；AMD EPYC CPU 亦同。
  - 結論著重功耗預算與 benchmark 方法（需考慮輸入熵），估計 FPU 能量上限 8 pJ/FLOP（GPU）、15 pJ/FLOP（CPU）。
  - **關係**：觀察同一現象，未利用它加速，也未處理精度。

## 2. 低精度與數值

- **[Gupta2015]** Suyog Gupta, Ankur Agrawal, Kailash Gopalakrishnan, Pritish Narayanan.
  *Deep Learning with Limited Numerical Precision.* ICML 2015, PMLR 37:1737–1746; arXiv:1502.02551.
  https://proceedings.mlr.press/v37/gupta15.html
  - 捨入方式決定低精度訓練成敗；16 位元定點＋隨機捨入幾乎無損。**關係**：LoRA 捨入方式對照（截斷／最近／隨機）的依據。

- **[Micikevicius2022]** Paulius Micikevicius, Dusan Stosic, et al. *FP8 Formats for Deep Learning.* arXiv:2209.05433.
  https://arxiv.org/abs/2209.05433
  - E4M3（權重／activation）與 E5M2（梯度）；配合 per-tensor scaling，master weight 與更新保留高精度。
  - **關係**：本研究 LoRA 截斷只作用在凍結的 FP8 基底權重，activation、梯度、更新與 scaling 維持此做法。

- **[Klower2021]** Milan Klöwer, Miha Razinger, Juan J. Dominguez, Peter D. Düben, Tim N. Palmer.
  *Compressing atmospheric data into its real information content.* Nature Computational Science 1, 713–724 (2021).
  https://www.nature.com/articles/s43588-021-00156-2
  - 以位元資訊量區分「真實資訊」與「雜訊」位元，CAMS 資料壓縮 17 倍保留 99% 資訊。
  - **關係**：「用到幾位 vs 需要幾位」的概念前例；目的為儲存壓縮而非運算功耗。

## 3. 混合精度求解與 FP64 模擬

- **[Haidar2018]** Azzam Haidar, Stanimire Tomov, Jack Dongarra, Nicholas J. Higham.
  *Harnessing GPU Tensor Cores for Fast FP16 Arithmetic to Speed up Mixed-Precision Iterative Refinement Solvers.*
  SC18, pp. 603–613. doi:10.1109/SC.2018.00050
  - **關係**：HPL-MxP 的演算法基礎（低精度 LU＋迭代修正）。

- **[Dongarra2025]** Jack Dongarra, Piotr Luszczek.
  *HPL-MxP Benchmark: Mixed-Precision Algorithms, Iterative Refinement, and Scalable Data Generation.*
  International Journal of High Performance Computing Applications (2025); arXiv:2509.19618. https://arxiv.org/abs/2509.19618

- **[Ootomo2024]** Hiroyuki Ootomo et al. *DGEMM on Integer Matrix Multiplication Unit.* arXiv:2306.11975.
  https://arxiv.org/abs/2306.11975 — Ozaki scheme 於 INT8 tensor core。

- **[Ozaki2025]** Katsuhisa Ozaki, Yuki Uchino, Toshiyuki Imamura.
  *Ozaki Scheme II: A GEMM-oriented emulation of floating-point matrix multiplication using an integer modular technique.*
  arXiv:2504.08009（2025）；IJHPCA（2026）。https://arxiv.org/abs/2504.08009
  - **關係**：rocHPL 分支的 GEMMul8 即基於此；屬「換運算格式」路線，與本研究「同格式內截斷尾數」互補。

- **[Uchino2025]** Yuki Uchino et al. *High-Performance and Power-Efficient Emulation of Matrix Multiplication using INT8
  Matrix Engines.* SC '25 Workshops. doi:10.1145/3731599.3767539（書目細節投稿前再核對）

- **[Huang2026]** Hua Huang, Wenkai Shao, Jeff Hammond.
  *Accelerating Density Fitting with Adaptive-precision and 8-bit Integer on AI Accelerators.* arXiv:2601.08077（2026）。
  https://arxiv.org/abs/2601.08077
  - 在 GPU4PySCF 中**依 SCF 迭代的相對能量變化選擇模擬精度**（INT8 Ozaki，3 到 6 個切片），能量誤差 < 1e-7 Eh；
    RTX 4090、RTX 6000 Ada 加速最多 204%、364%，H100 37%。**未討論功耗、時脈或功耗上限。**
  - **關係（重要）**：「依收斂程度調整精度」已有前例；本研究 M2 的差異在於目標為功耗受限下的時脈（FP64 單元不變、
    以尾數截斷取代格式模擬）、應用為平面波 DFT（QE）、訊號為 Davidson 門檻 ethr。論文中必須引用並明確區分。

## 4. 應用軟體

- **[Giannozzi2020]** P. Giannozzi et al. *Quantum ESPRESSO toward the exascale.* J. Chem. Phys. 152, 154105 (2020).
  https://pubs.aip.org/aip/jcp/article/152/15/154105/1058748/
- **[Wu2024]** Xiaojie Wu et al. *Enhancing GPU-acceleration in the Python-based Simulations of Chemistry Framework*
  （GPU4PySCF）。arXiv:2404.09452. https://arxiv.org/abs/2404.09452

## 待補

- MLPerf Training 規則與 Llama2-70B LoRA 參考實作（mlcommons/training_policies、mlcommons/training）的正式引用格式。
- 投稿前以 Google Scholar／IEEE Xplore／ACM DL 做一次系統性回顧（關鍵字：data-dependent power、bit toggling、
  mantissa truncation、precision tuning energy、power capping HPC），特別是 SC、ISC、IPDPS、HPDC 近兩年。

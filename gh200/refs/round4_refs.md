# 第四輪參考文獻查證（審稿人要求補的引用）

查證日期：2026-09-30。每筆都寫出實際讀到的來源。查證狀態分三種：

- **已核對原始頁面**：讀過出版社頁、arXiv、專利頁、官方文件或作者公開的全文。
- **書目已核對，內容只有二手／摘要**：DOI 登錄資料（Crossref，由出版社提交）確認了書目，但出版社頁（IEEE Xplore／ACM DL／ACS）擋掉自動讀取，內容只能靠摘要或第三方描述。
- **找不到**：本輪沒有這種情況。

引號內是原文。「（我方計算）」是我從原文數字自己算出來的，原文沒有這個數字。

---

## 1. Bhalachandra2022

**書目**：Sridutt Bhalachandra, Brian Austin, Samuel Williams, Nicholas J. Wright. *Understanding the Impact of Input Entropy on FPU, CPU, and GPU Power.* arXiv:2212.08805 [cs.DC], v1 2022-12-17. https://arxiv.org/abs/2212.08805 （doi:10.48550/arXiv.2212.08805）。arXiv 頁沒有 journal-ref，Crossref 也查不到後續的正式出版版本，所以只能引用 arXiv。

**重點（讀過 arXiv PDF 全文）**：
- A100 上做 16384×16384 DGEMM（100 次），各元素相同值（fixed）和隨機值（random）兩種輸入相比：「The average power consumed by the random case (≈398 W) is about 67% higher than the fixed case (≈239 W)」，而且 random 的效能低了將近 4%（18.6 vs 19.4 TFLOP/s）。
- 結論：「GPU DGEMM power is shown to be highly dependent on the entropy of inputs to the FPUs ranging from roughly 60-100% of TDP」。論文還估出 FPU 能量上限：GPU 8 pJ/FLOP、CPU 15 pJ/FLOP。

**查證狀態**：已核對原始頁面（arXiv 摘要頁與 PDF）。

---

## 2. Gregersen2024

**書目**：Theo Gregersen, Pratyush Patel, Esha Choukse. *Input-Dependent Power Usage in GPUs.* In SC24-W: Workshops of the International Conference for High Performance Computing, Networking, Storage and Analysis, IEEE, pp. 1872–1877, 2024. doi:10.1109/SCW63240.2024.00235。預印本 arXiv:2409.18324（v1 2024-09-26）。

**重點（讀過 arXiv PDF 全文）**：
- **有刻意避開降頻**：「All experiments use 2048 by 2048 matrices. We selected 2048 as the largest power of two that did not consistently throttle the A100 GPU.」（§III）。推廣實驗中，RTX 6000 改用 512×512，因為「it throttled at 2048 by 2048」（§IV-E）。論文沒有鎖時脈、沒有設功耗上限，也沒有量「功耗上限下的效能」。
- **只測 GEMM**：只用「standard NVIDIA CUTLASS kernels optimized for GEMM execution」，功耗由 dcgm 每 100 ms 取樣一次。主要平台是 Azure A100 PCIe（TDP 300 W），部分實驗在 H100、V100、RTX 6000 上重現。資料型別有 FP32、FP16、FP16-T（tensor core）、INT8。
- 摘要寫功耗變化「almost 40%」，引言寫「up to 38%」。和本研究直接相關的兩條是 T5「As more least significant bits are randomized, power increases」和 T14「Zeroing least significant bits can reduce power」。成因（bit flips）只列為假說，驗證留待未來工作。

**查證狀態**：已核對原始頁面（arXiv PDF 全文；正式出版的頁碼與 DOI 經 Crossref 核對）。

---

## 3. He2024（Horace He 部落格）

**書目**：Horace He. *Strangely, Matrix Multiplications on GPUs Run Faster When Given "Predictable" Data! [short].* Thonk From First Principles（Substack），2024-04-29。副標題：「Great minds discuss flops per watt.」 https://www.thonking.ai/p/strangely-matrix-multiplications

**日期注意**：某個搜尋引擎摘要寫成 2024-08-07，這是錯的。頁面本身顯示「Apr 29, 2024」，最早的讀者留言（Dylan Patel）也是 Apr 29, 2024。

**他示範了什麼（讀過頁面全文）**：
- 起因是 8192×8192×8192 matmul 在 PyTorch 呼叫 cuBLAS 得到 258 TFLOPS，CUTLASS profiler 卻得到 288 TFLOPS。追查後發現 CUTLASS profiler 預設「only initializes the inputs with integers」。
- 關鍵示範：`zero_inputs = torch.zeros(N, N)` → 「295 Teraflops」，`randn_inputs = torch.randn(N, N)` → 「257 Teraflops」，零輸入快約 14.8%（我方計算：295/257）。**用的卡是 A100**：腳註寫「The A100 GPU I'm testing this on happens to have a power limit of 330W.」頁面文字與可見程式碼都沒寫資料型別（dtype），不要自行補上。
- 功耗上限掃描：「as the power limit decreases from 330W down to 100W (the minimum), the relevant performance improvement from using predictable inputs increases. Interestingly, at the lowest power limit (100W), the trend reverses.」他還嘗試了棋盤格、稀疏、uniform 等分佈。解釋採用 dynamic/switching power（power ≈ clock × transistor flips）。結論是「matmuls on the H100 are primarily not compute or bandwidth limited, they are power limited.」H100 部分是理論峰值與時脈的討論，零／隨機輸入的實測是在 A100 上做的。

**查證狀態**：已核對原始頁面。

---

## 4. AMD2022（美國專利 US 11,455,766 B2）

**書目**：Pramod V. Argade, Daniel Nikolai Peroni（受讓人：Advanced Micro Devices, Inc.）. *Variable precision computing system.* US Patent 11,455,766 B2，申請號 US 16/134,856，申請日（亦為優先權日）2018-09-18，早期公開 US 2020/0090397 A1（2020-03-19），**公告日 2022-09-27**。https://patents.google.com/patent/US11455766B2/en

**摘要重點（讀過 Google Patents 頁面的摘要、說明書與請求項）**：
- 摘要：GPU shader 等功能單元內有 zeroing module，依 precision flag 把輸入或輸出運算元的指定部分寫成 0。說明書寫「Because a portion of the data has been zeroed, the functional unit consumes less power during data processing.」
- 說明書明確寫到歸零的是尾數：「zero a portion of the mantissa component of each data operand, without modifying either of the sign component and the exponent component」。省電的理由是位元在 0 和 1 之間變化時，正反器與邏輯會耗電。precision flag 可以依電源管理設定（例如高效能模式）調整。
- 請求項 1 是一個方法：依 shader 的 flag 狀態，把資料的一部分替換成 0 後再運算。這是**硬體內建**的歸零模組，應用在圖形 shader、以使用者感知品質為準；本研究是在 GEMM 之前用純軟體歸零，目標是 HPC 數值正確性。

**查證狀態**：已核對原始頁面（Google Patents）。

---

## 5. Tong2000

**書目**：Jonathan Ying Fai Tong, David Nagle, Rob A. Rutenbar. *Reducing power by optimizing the necessary precision/range of floating-point arithmetic.* IEEE Transactions on Very Large Scale Integration (VLSI) Systems, 8(3):273–286, June 2000. doi:10.1109/92.845894

**重點（僅摘要）**：摘要寫「Most FP programs in our benchmark suite maintain the same output even when the mantissa bitwidth is reduced by half」，以及「up to 66% reduction in multiplier energy/operation can be achieved in the FP unit by this bitwidth reduction technique without sacrificing any program accuracy」。做法是使用可變位寬的硬體 FP 單元，測試程式處理的是低解析度的人類感官資料（影音等）。

**查證狀態**：書目已核對（Crossref DOI 資料：卷 8、期 3、頁 273–286、2000 年 6 月），摘要讀自 University of Illinois 的 Illinois Experts 機構頁。IEEE Xplore 原頁擋掉自動讀取（HTTP 418），全文需要學校網路。

---

## 6. Pool2008、Pool2011

### Pool2008

**書目**：Jeff Pool, Anselmo Lastra, Montek Singh. *Energy-precision tradeoffs in mobile graphics processing units.* In 2008 IEEE International Conference on Computer Design (ICCD), pp. 60–67, October 2008. doi:10.1109/ICCD.2008.4751841

**重點（只有二手描述）**：作者後來的 UNC 技術報告 TR10-017 把這篇描述為：為 GPU 的 vertex transformation stage 建立誤差模型，「Our prior analytical model suggested that we could realize savings of around 35% [1]」，這裡的 [1] 就是本篇。作者 HPG 2011 投影片把它列為「Variable-precision vertex shaders – Pool, Singh, and Lastra, 2008」。

**查證狀態**：書目已核對（Crossref DOI 資料）。內容只有二手來源（作者自己的後續技術報告與投影片），IEEE Xplore 擋掉自動讀取，摘要與全文需要學校網路。

### Pool2011

**書目**：Jeff Pool, Anselmo Lastra, Montek Singh. *Precision selection for energy-efficient pixel shaders.* In Proceedings of the ACM SIGGRAPH Symposium on High Performance Graphics (HPG '11), pp. 159–168, 2011. doi:10.1145/2018323.2018349

**重點**：作者在 HPG 官網公開的投影片（https://www.highperformancegraphics.org/previous/www_2011/media/Papers/HPG2011_Papers_Pool.pdf）寫著「Full Precision (24 bits in mantissa, 100% energy)」對「Reduced Precision (12.5 bits in mantissa, 29% energy)」。programmer-directed 方式在算術上省「70-80%」、整體 GPU 省「10-20%」，能量是修改 ATTILA 模擬器估出來的，不是硬體實測。多個搜尋結果轉述的摘要數字是「pixel shader 算術能量省 71%、GPU 整體最多省 20%」，但我沒有讀到論文本身，這組數字屬二手。

**查證狀態**：書目已核對（Crossref DOI 資料）。內容讀自作者公開的投影片，論文全文在 ACM DL（HTTP 403），需要學校網路。

---

## 7. Guenther2016

**書目**：D. Guenther, T. Henriksson, R. Leupers, G. Ascheid. *Mantissa-masking for energy-efficient floating-point LTE uplink MIMO baseband processing.* In Proceedings of the 2016 Design, Automation & Test in Europe Conference & Exhibition (DATE), pp. 1028–1029, Dresden, 2016. doi:10.3850/9783981537079_0389。PDF：https://past.date-conference.com/proceedings-archive/2016/pdf/0389.pdf

作者單位：RWTH Aachen（Guenther、Leupers、Ascheid）、Huawei Technologies Sweden（Henriksson）。這篇是兩頁的 industry case study。

**重點（讀過 DATE 官方 PDF 全文）**：
- 定義：「mantissa-masking (setting a number of least significant bits to zero) exploits this fact to reduce switching activity and thereby energy consumption」。實作是「applying a configurable bitmask to the mantissa of each input of each arithmetic unit」，處理器使用 12 位元尾數格式，核心在 TSMC 28 nm 合成，遮罩的硬體開銷可忽略。
- 結果（Table I，2×4 MIMO）：每個任務最多省能 23%（QRD、4-QAM 23.3%；LDLh 4-QAM 15.1%）。記憶體不受遮罩影響，佔功耗 15.8–17.2%。文中沒有說明數字來自矽晶片實測還是合成後模擬。

**查證狀態**：已核對原始頁面（DATE 官方 PDF；DOI 經 Crossref 核對）。

---

## 8. Jahadi2024

**書目**：Reza Jahadi, Ehsan Atoofian. *Low-Power Register File for Tensor Cores.* In 2024 IEEE 15th International Green and Sustainable Computing Conference (IGSC), pp. 159–164, 2024. doi:10.1109/IGSC64514.2024.00037

Crossref 上兩人在 IGSC 2024 只有這一篇。有個搜尋摘要說它收錄在 LNCS vol. 14802，這是錯的；它是 IEEE 出版。

**重點（僅摘要）**：
- 問題是 tensor core 暫存器檔的**漏電（leakage）功耗**。觀察到「register bits exhibit a strong bias towards zeros」，據此設計低功耗 SRAM cell（LPS）：「In the preferred state, the leakage power is smaller by as much as approximately 49×」。
- 延伸設計：LPS+ 選擇性反轉位元；PLSP+「drops low-order bits in network values」，利用 CNN 的容錯性。整體「leakage power is reduced by 77.3% with a negligible impact on accuracy」。
- 與本研究的差異：這篇是**需要新 SRAM cell 的硬體提案**，省的是**靜態漏電**；本研究是純軟體，省的是 FPU 與資料路徑的**動態翻轉功耗**，並在現有 GH200 上實測。

**查證狀態**：書目已核對（Crossref DOI 資料）。摘要取自 Semantic Scholar 索引（IEEE Xplore 擋掉自動讀取），全文需要學校網路。

---

## 9. Ujeniya2026

**書目**：Aditya Ujeniya, Jan Eitzinger, Georg Hager, Gerhard Wellein（Erlangen National High Performance Computing Center, NHR@FAU, Friedrich-Alexander-Universität Erlangen-Nürnberg）. *Architectural Trade-offs in the Energy-Efficient Era: A Comparative Study of power-capping NVIDIA H100 and H200.* arXiv:2604.11391 [cs.PF]，v1 2026-04-13，v2 2026-07-15。https://arxiv.org/abs/2604.11391 。arXiv 頁沒有 journal-ref。

**約 100 W 的數字在哪裡（讀過 v1、v2 PDF）**：在 **§3.5 Methodology**（v2 PDF 第 6 頁）。v2 原文：「All experiments use random data initialization to maximize power draw. Randomly initialized data causes more dynamic power due to more frequent transistor switching [1,6]; we experimented with different array initialization techniques like constant and random data and found up to 100W difference in power draw. This is in line with similar studies on recent CPUs and GPUs [9,2], but it is not within the scope of this study to go into detail of this behavior. More data on this topic is available online.」v1 同一節有幾乎相同的句子（寫成「100 W」）。

**注意**：
- 原文沒有說這 100 W 是在哪張卡（H100 或 H200）、哪個 benchmark（DGEMM 或 TheBandwidthBenchmark）量到的，也沒有給數據表。
- 腳註指向的部落格 https://blogs.fau.de/adityauj/2025/11/15/power-frequency-characteristics/ 今天回傳 HTTP 410 Gone，已無法讀取。
- 引用 [2] 是 Antepara et al. SC'25（第 10 筆）。論文主體是 200–700 W 功耗上限下 H100 與 H200 的 DGEMM、TBB、ViT 比較。

**查證狀態**：已核對原始頁面（arXiv 摘要頁與 v1、v2 PDF）。

---

## 10. Antepara2025W、Antepara2025

### Antepara2025W（SC25 Workshops）

**書目**：Oscar Antepara, Leonid Oliker, Samuel Williams. *Roofline Analysis of Tightly-Coupled CPU-GPU Superchips: A Study on MI300A and GH200.* In Proceedings of the SC '25 Workshops of the International Conference for High Performance Computing, Networking, Storage and Analysis (SC Workshops '25), St. Louis, MO, ACM, pp. 1205–1216, 2025. doi:10.1145/3731599.3767497。在 P3HPC 2025 workshop 發表。

**開放全文**：CC BY 4.0，eScholarship（LBNL）：https://escholarship.org/uc/item/3zr637h8 ，**不需要學校網路**。

**有沒有量資料相依功耗（讀過全文）**：
- **沒有直接報告功耗數字**，量的是「zeros 對 random 輸入」下的效能與頻寬（Mixbench 向量 FP64）。作者修改了 Mixbench，讓輸入落在 1.0–2.0、常數設為 −2.0，避免乘加結果趨向 0 而壓低功耗。
- **GH200**：「For the NVIDIA GH200, there is no significant difference in peak values when using zeros or random values as input data.」（§5.2；Table 4 標題也寫「did not show any difference」）。
- **MI300A**：random 輸入讓頻寬降約 14%、效能降約 27%，歸因於 DVFS 在 550 W TDP 下調降。結論段：「randomized input data demands substantially more energy per bit or energy per FLOP than zero data. For power-constrained superchips, higher energy per bit or FLOP results in lower bandwidth and performance for randomized input data.」

### Antepara2025（SC25 主會）

**書目**：Oscar Antepara, Zhengji Zhao, Brian Austin, Nan Ding, Leonid Oliker, Nicholas J. Wright, Samuel Williams. *Benchmark-driven Models for Energy Analysis and Attribution of GPU-Accelerated Supercomputing.* In Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (SC '25), St. Louis, MO, ACM, pp. 888–904, 2025. doi:10.1145/3712285.3759815

**開放全文**：CC BY 4.0，eScholarship：https://escholarship.org/uc/item/6189368s ，**不需要學校網路**。

**有沒有量資料相依功耗（讀過全文）**：**有，而且包含 GH200。**
- **Table 2**（GEMM power measurement and performance for zeros or random input data）中的 GH200 數字：
  - DGEMM：zeros 563 W／65.4 TF/s，random 898 W／57.8 TF/s（我方計算：功耗 +59.5%，效能 −11.6%）
  - SGEMM（NVIDIA 上用 TF32）：zeros 663 W／432 TF/s，random 863 W／386 TF/s
  - HGEMM：zeros 578 W／875 TF/s，random 854 W／757 TF/s
  - Pconst 72 W
  
  測試機是 NERSC 的 ATE testbed，「the NVIDIA GH200 has a TDP of 900W」。功耗用 nvidia-smi 每 1 秒取樣。
- §5.1：「Almost universally, randomized inputs push GPU power to the TDP limit resulting in a commensurate reduction in performance as the GPUs reduce frequency.」§5.2：「On the A100, GH200, and MI250X, computation on zeros never reaches TDP. As such, clock rates are never capped, and performance and bandwidth are boosted.」
- 方法（§4.1）：用 random 輸入量「control+datapath」能量，用 zeros 輸入只量「control」能量，兩者相減得到 datapath 能量，這個做法沿用 Bhalachandra et al.。相關工作段指出「Existing GPU power models often overlook data-dependent power variation」，並把這個現象的首次觀察歸給 Lucas et al.（第 12 筆），再由 Bhalachandra 與 Gregersen 確認。
- 對本研究的意義：這是**已發表、同為 GH200** 的 DGEMM 零對隨機功耗與效能實測。本論文應引用它，並說明差異：本研究處理的是隨機資料裡的「部分尾數位元」，且驗證了數值結果。

**查證狀態**：兩篇都已核對原始頁面（eScholarship 全文 PDF；頁碼與 DOI 經 Crossref 核對）。

---

## 11. TensorRTDocs

**建議引用（與「資料值、功耗、降頻」最直接相關）**：NVIDIA. *Performance Benchmarking — NVIDIA TensorRT*（TensorRT 11.x 文件，§「GPU Power Consumption and Power Throttling」）。https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/benchmarking.html （頁尾「Last updated on Sep 08, 2026」，查閱日 2026-09-30）

- 原文：「Lastly, power consumption can depend on the activation values, causing different input performance measurements. For example, if all the network input values are set to zeros or NaNs, the GPU consumes less power than when the inputs are normal values because of fewer bit-flips in DRAM and the L2 cache. To avoid this discrepancy, always use the input values that best represent the actual value distribution when measuring the performance.」
- 同一節也寫：「Power throttling occurs when the average GPU power consumption reaches the power limit ... the driver has to throttle the clock to a lower frequency」。
- 如果要固定版本，10.x 分支有同樣文字：*Hardware/Software Environment for Performance Measurements — NVIDIA TensorRT*，https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/performance/hw-sw-environment.html （Last updated May 23, 2026）。

**如果要引用「降低精度換取效能／能效」**：NVIDIA. *Working with Quantized Types — NVIDIA TensorRT*。https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-with-quantized-types.html 。原文：「quantization, a technique that reduces model size and accelerates computation by representing floating-point values with lower-precision data types」，Key Benefits 列出「Improves energy efficiency」。支援的格式包括 FP8E4M3（4 位指數、3 位尾數）。

**查證狀態**：已核對原始頁面（NVIDIA 官方文件）。

---

## 12. Lucas2016

**書目**：Jan Lucas, Ben Juurlink（TU Berlin）. *ALUPower: Data Dependent Power Consumption in GPUs.* In 2016 IEEE 24th International Symposium on Modeling, Analysis and Simulation of Computer and Telecommunication Systems (MASCOTS), pp. 95–104, September 2016. doi:10.1109/MASCOTS.2016.21。作者接受稿（開放取用）：https://doi.org/10.14279/depositonce-7074

**對「資料值／翻轉影響功耗」的具體發現（讀過作者接受稿全文）**：
- GTX 580（Fermi）上跑 FMUL 測試核心（Fig. 1）：全 0 輸入 155 W，隨機輸入 222 W，全 0 與全 1 交替（每個位元每週期都翻轉）257 W。內文寫最後一種比全 0 高「65.8%」（我方計算：257/155 = 1.658，一致）。**注意**：內文說隨機輸入「results in an increase of 77 W」，和圖上 222 W（差 67 W）不一致，引用時建議用圖中數字或 65.8%。結論列出「the large (≥ 30%) influence of data values on the energy consumption on both Fermi and Maxwell GPUs」（GTX 580、GTX 750 Ti）。
- **和「低位填 0／填 1 對照」直接相關的段落**（§V-B）：作者原本以為常數輸入不會讓訊號變化，所以功耗與常數值無關，但實測「this assumption is not true」。能耗會隨所用的常數值（set bits 數，即 POPC）以及暫存器 bank 而變；在 even warps 中，有些非零常數向量甚至比全 0 基準**更省**（「if some wires or gates are charged to 1 first and need to be recharged only if a 0 is transmitted」）。
- 他們的巨觀模型（Eq. 1）以連續運算元之間的 **Hamming distance（翻轉數）** 為主要項，**POPC（1 的個數）** 是額外的一個次要項。所以文獻上支持「翻轉是主因、1 的個數是次要效應」，可以用來佐證本研究填 1 對照組的設計。

**查證狀態**：已核對原始頁面（作者接受稿 PDF，TU Berlin DepositOnce；書目經 Crossref 核對）。

---

## 13. Dawson2024

**書目**：William Dawson, Katsuhisa Ozaki, Jens Domke, Takahito Nakajima. *Reducing Numerical Precision Requirements in Quantum Chemistry Calculations.* Journal of Chemical Theory and Computation, 20(24):10826–10837, 2024（線上 2024-12-07）。doi:10.1021/acs.jctc.4c00938。預印本 arXiv:2407.13299（v2 2024-10-22）。

**重點（讀過 arXiv v2 全文）**：
- 研究的 kernel 是用密度矩陣純化（purification）從 HF／DFT Hamiltonian（LCAO 基底）求單粒子密度矩陣。用 MPFR 模擬 11 到 53 位的尾數：「we found that double precision affords an unnecessarily high level of precision. If a precision between single and double precision is used (for example, with an effective 37 bit mantissa), a reliable result can be obtained.」並發現「37 bits of effective precision is sufficient for accumulation」。
- 提出用較少切片的 Ozaki scheme，在 FP16 tensor core 上約省一半乘法量。測試用的 Hamiltonian 來自 **PySCF**、BigDFT、NTChem。
- **沒有討論功耗、時脈或功耗上限**（全文搜尋 power consumption、watt、energy efficiency 都沒有結果），它的目標是低精度硬體的運算吞吐量。

**查證狀態**：已核對原始頁面（arXiv 全文；正式出版書目經 Crossref 核對；ACS 頁擋掉自動讀取）。

---

## 14. Woo2023

**書目**：Jeheon Woo, Seonghwan Kim, Woo Youn Kim（KAIST 化學系）. *Dynamic Precision Approach for Accelerating Large-Scale Eigenvalue Solvers in Electronic Structure Calculations on Graphics Processing Units.* Journal of Chemical Theory and Computation, 19(5):1457–1465, 2023（線上 2023-02-22）。doi:10.1021/acs.jctc.2c00983，PMID 36812094。

**做法（僅摘要，來自 PubMed；ACS 全文被擋，也找不到預印本）**：
- **切換什麼**：「SP, DP, and mixed precision are dynamically switched during an iterative diagonalization process」，也就是在 FP32、FP64 與混合精度三種**格式**之間切換。
- **用什麼訊號切換**：切換發生在 **LOBPCG 特徵值求解器**內部。門檻是事先定好的：「We determined a proper threshold for switching each precision scheme by examining the convergence pattern on the eigenvalue solver only with the kinetic energy operator of the Kohn-Sham Hamiltonian.」**依據是特徵求解器的收斂狀態，不是直接看 SCF 收斂程度**，本論文描述時要改正這一點。摘要沒寫收斂量是殘差範數還是別的，需要讀全文確認。
- **結果**：「up to 8.53× and 6.60× speedups for band structure and self-consistent field calculations」，硬體只說「on NVIDIA GPUs」，摘要沒寫型號。
- **程式**：摘要**沒有寫**。Woo 與 W. Y. Kim 是實空間程式 ACE-Molecule（J. Chem. Phys. 2020, doi:10.1063/5.0002959）的共同作者，但這篇是否用 ACE-Molecule 必須讀全文才能確認，**不可直接寫入論文**。
- **功耗**：摘要沒提功耗或能源。全文是否提到不確定，需要學校網路下載確認。

**與本論文的具體差異**：
1. 目標不同：Woo 靠 GPU 上 FP32 吞吐量遠高於 FP64 取得加速，靠的是換運算單元。本論文**始終使用同一個 FP64 單元**，理論峰值不變，而是讓位元翻轉減少、功耗下降，在**功耗上限下換得更高時脈**。
2. 調整的東西不同：Woo 在 SP、DP、混合三種離散格式間切換。本論文維持 FP64 格式，只調整保留的尾數位數，是位元層級、幾乎連續可調的截斷。
3. 切換訊號不同：Woo 看 LOBPCG 的收斂狀態，門檻用只含動能算子的 Hamiltonian 事先標定。本論文依 Quantum ESPRESSO Davidson 對角化的收斂門檻 ethr 決定保留幾個尾數位元；ethr 在 QE 中本來就會隨 SCF 進度收緊。
4. 求解器與程式不同：LOBPCG（程式未明，可能是實空間基底）對 QE 的 Davidson（平面波基底），本論文另外還驗證了 PySCF 與 HPL。
5. 功耗：摘要未涉及。本論文以功耗、時脈、吞吐量作為主要量測結果。

**查證狀態**：書目已核對（Crossref＋PubMed）。內容僅摘要，ACS 全文需要學校網路。

---

## 15. Altera2017（美國專利 US 9,606,608 B1）

**書目**：Martin Langhammer（受讓人：Altera Corporation, San Jose, CA）. *Low power optimizations for a floating point multiplier.* US Patent 9,606,608 B1，申請號 14/219,421，申請日 2014-03-19，**公告日 2017-03-28**。https://patents.google.com/patent/US9606608B1/en ，PDF：https://patentimages.storage.googleapis.com/36/2c/6a/d93a843b1aff75/US9606608.pdf

**那句話的原文**：在說明書**第 8 欄**。這是依 PDF 文字層和欄邊行號推算的，約在第 30–46 行，投稿前請對照 PDF 第 13 頁確認。
> 「One way to reduce power in a multiplier is to zero out (or truncate) a number of the least significant bits of the inputs into the multiplier. However, performing this type of truncation at the input stage has barely any effect on reducing the amount of power consumed by the device. In an example, the least significant 6 bits of both inputs into a 27×27 multiplier as shown in FIG. 3 are truncated (such that only the most significant 21 bits remain in each input). In this case, 60% of the partial products in the 27×9 multiplier at 334 and in the 18×9 multiplier at 336 are zeroed, thereby reducing some power. However, a large portion of the compressor tree is still operating. Because the compressor tree may consume more power than the partial product generator and a large proportion of the overall power in the processing block, the amount of power reduction when simply truncating at the inputs of a multiplier is not significant.」

**引用時的背景**：
- 講的是可程式化 IC（FPGA）DSP block 中，由小乘法器組成的 27×27 定點乘法器。這是**定性陳述，沒有量測數字**。
- 專利的解法是用 switching circuitry 關掉負責低位的子乘法器（見摘要）。
- 本論文在 GH200 上實測到明顯省電，和它並不矛盾：可以寫成「在該 FPGA 乘法器架構下，只在輸入端歸零效果有限，因為 compressor tree 仍在運作」，並指出 GPU tensor core／FMA 資料路徑上的實測結果不同。

**查證狀態**：已核對原始頁面（Google Patents 全文與 USPTO 專利 PDF 首頁；受讓人、日期、欄位都看過）。

---

## 可直接貼進論文的書目清單

```
[Bhalachandra2022] Sridutt Bhalachandra, Brian Austin, Samuel Williams, Nicholas J. Wright. Understanding the Impact of Input Entropy on FPU, CPU, and GPU Power. arXiv preprint arXiv:2212.08805, 2022. https://arxiv.org/abs/2212.08805
[Gregersen2024] Theo Gregersen, Pratyush Patel, Esha Choukse. Input-Dependent Power Usage in GPUs. In SC24-W: Workshops of the International Conference for High Performance Computing, Networking, Storage and Analysis, pp. 1872–1877, IEEE, 2024. doi:10.1109/SCW63240.2024.00235
[He2024] Horace He. Strangely, Matrix Multiplications on GPUs Run Faster When Given "Predictable" Data! [short]. Thonk From First Principles (blog), April 29, 2024. https://www.thonking.ai/p/strangely-matrix-multiplications
[AMD2022] Pramod V. Argade, Daniel Nikolai Peroni (Advanced Micro Devices, Inc.). Variable precision computing system. US Patent 11,455,766 B2, filed Sep. 18, 2018, issued Sep. 27, 2022. https://patents.google.com/patent/US11455766B2/en
[Tong2000] Jonathan Ying Fai Tong, David Nagle, Rob A. Rutenbar. Reducing power by optimizing the necessary precision/range of floating-point arithmetic. IEEE Transactions on Very Large Scale Integration (VLSI) Systems, 8(3):273–286, 2000. doi:10.1109/92.845894
[Pool2008] Jeff Pool, Anselmo Lastra, Montek Singh. Energy-precision tradeoffs in mobile graphics processing units. In 2008 IEEE International Conference on Computer Design (ICCD), pp. 60–67, 2008. doi:10.1109/ICCD.2008.4751841
[Pool2011] Jeff Pool, Anselmo Lastra, Montek Singh. Precision selection for energy-efficient pixel shaders. In Proceedings of the ACM SIGGRAPH Symposium on High Performance Graphics (HPG '11), pp. 159–168, 2011. doi:10.1145/2018323.2018349
[Guenther2016] D. Guenther, T. Henriksson, R. Leupers, G. Ascheid. Mantissa-masking for energy-efficient floating-point LTE uplink MIMO baseband processing. In Design, Automation & Test in Europe Conference & Exhibition (DATE), pp. 1028–1029, 2016. doi:10.3850/9783981537079_0389
[Jahadi2024] Reza Jahadi, Ehsan Atoofian. Low-Power Register File for Tensor Cores. In 2024 IEEE 15th International Green and Sustainable Computing Conference (IGSC), pp. 159–164, 2024. doi:10.1109/IGSC64514.2024.00037
[Ujeniya2026] Aditya Ujeniya, Jan Eitzinger, Georg Hager, Gerhard Wellein. Architectural Trade-offs in the Energy-Efficient Era: A Comparative Study of power-capping NVIDIA H100 and H200. arXiv preprint arXiv:2604.11391, 2026. https://arxiv.org/abs/2604.11391
[Antepara2025W] Oscar Antepara, Leonid Oliker, Samuel Williams. Roofline Analysis of Tightly-Coupled CPU-GPU Superchips: A Study on MI300A and GH200. In Proceedings of the SC '25 Workshops of the International Conference for High Performance Computing, Networking, Storage and Analysis, pp. 1205–1216, ACM, 2025. doi:10.1145/3731599.3767497
[Antepara2025] Oscar Antepara, Zhengji Zhao, Brian Austin, Nan Ding, Leonid Oliker, Nicholas J. Wright, Samuel Williams. Benchmark-driven Models for Energy Analysis and Attribution of GPU-Accelerated Supercomputing. In Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (SC '25), pp. 888–904, ACM, 2025. doi:10.1145/3712285.3759815
[TensorRTDocs] NVIDIA. Performance Benchmarking — NVIDIA TensorRT (section "GPU Power Consumption and Power Throttling"). NVIDIA TensorRT Documentation, 2026 (accessed Sep. 30, 2026). https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/benchmarking.html
[TensorRTQuant] NVIDIA. Working with Quantized Types — NVIDIA TensorRT. NVIDIA TensorRT Documentation, 2026 (accessed Sep. 30, 2026). https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-with-quantized-types.html
[Lucas2016] Jan Lucas, Ben Juurlink. ALUPower: Data Dependent Power Consumption in GPUs. In 2016 IEEE 24th International Symposium on Modeling, Analysis and Simulation of Computer and Telecommunication Systems (MASCOTS), pp. 95–104, 2016. doi:10.1109/MASCOTS.2016.21
[Dawson2024] William Dawson, Katsuhisa Ozaki, Jens Domke, Takahito Nakajima. Reducing Numerical Precision Requirements in Quantum Chemistry Calculations. Journal of Chemical Theory and Computation, 20(24):10826–10837, 2024. doi:10.1021/acs.jctc.4c00938
[Woo2023] Jeheon Woo, Seonghwan Kim, Woo Youn Kim. Dynamic Precision Approach for Accelerating Large-Scale Eigenvalue Solvers in Electronic Structure Calculations on Graphics Processing Units. Journal of Chemical Theory and Computation, 19(5):1457–1465, 2023. doi:10.1021/acs.jctc.2c00983
[Altera2017] Martin Langhammer (Altera Corporation). Low power optimizations for a floating point multiplier. US Patent 9,606,608 B1, filed Mar. 19, 2014, issued Mar. 28, 2017. https://patents.google.com/patent/US9606608B1/en
```

## 仍需學校網路才能補齊的部分

- Tong2000、Pool2008（IEEE Xplore）、Pool2011（ACM DL）、Jahadi2024（IEEE Xplore）的全文。目前只讀到摘要或作者投影片、技術報告。
- Woo2023（ACS）全文，需要確認四件事：收斂訊號的確切定義、使用的程式（是否為 ACE-Molecule）、GPU 型號、有沒有提到功耗。
- Altera2017 的欄位與行號，請對照 PDF 第 13 頁（第 7–8 欄那一頁）確認「col. 8, ll. ~30–46」。

## 2026-10-03 全文核對（使用者以學校網路取得，PDF 在 gh200/refs/pdf/，不進 git）

- Tong2000：摘要「up to 66% reduction in multiplier energy/operation」→ 論文與報告寫法相符（乘法器每次運算的能量）。
- Pool2008：行動 GPU 頂點轉換階段省 23%（畫質幾乎不變）到 36%（輕微失真），以能耗模型估計。先前把 71% 一併歸給 Pool2008 是錯的，已改。
- Pool2011：一張影像的像素著色器算術省 71%，約 GPU 整體能耗 10–20%；靜態／動態方法 31%／70%／62%。
- Jahadi2024：利用暫存器位元偏向 0 的低漏電 SRAM，加上捨去 CNN 低位位元（PLSP+），漏電少 77.3%（模擬）。論文已補上。
- Woo2023：DynP 在能帶與 SCF 計算最多 8.53× 與 6.60×，「最多 8.5 倍」相符。

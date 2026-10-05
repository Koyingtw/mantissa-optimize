# 第五輪參考文獻查證：地統計／高斯過程 Cholesky 的降精度、cuBLAS FP64 模擬、GPU4PySCF DF-MP2

查證日期：2026-10-05。目的：論文新增的高斯過程（GP）最大概似實驗（Matérn 共變異、分塊 Cholesky、尾端更新為 FP64 DGEMM，面板尾數截斷到 28 位、功耗上限下快 5.5%）投稿前，要先正確引用並區分地統計混合精度的前人研究。

查證狀態（沿用第四輪）。所有卷期頁與 DOI 都經 Crossref 核對；引號內是原文；標「我方判讀」「我方推論」「我方計算」的是我們自己的解讀、推論或計算。
- **已核對原始頁面**：讀過全文（出版社版、作者或機構公開的 PDF、arXiv）或官方文件。
- **書目已核對，內容僅摘要**：Crossref（出版社提交的 DOI 資料）確認了書目，內容只讀到摘要。
- **UNVERIFIED**：沒有核對到原始來源，不可直接寫進論文。

## 0. 結論（會改變論文寫法的部分）

1. KAUST／ICL 這條線（ExaGeoStat → HiPC19 → TPDS22 → SC22 → CLUSTER23 → SC24 → ISC25）**全部是換格式**：依「離對角線的距離（band）」或「tile 的 Frobenius 範數比」，把整個 tile 轉成 FP32／FP16／FP8，或用 FP16 輸入、FP32 累加。對角 tile 留在 FP64（ISC25 原文沒有明講）。**沒有一篇在 FP64 格式內截斷尾數並保留 FP64 指數。**
2. **功耗不是空白**：Cao et al.（CLUSTER 2023）在單張 V100／A100／H100 上，用 NVIDIA 工具量了混合精度 Cholesky 的功耗、能耗與 Gflops/W。所以論文不能寫「首次在地統計混合精度中量 GPU 功耗」。能寫的是：這條線沒有人設功耗上限、沒有人分析時脈。
3. **「留 FP64 指數、只丟尾數」本身不是新點子**：CPMS（Grützmacher et al., ACM TOPC 2020）的摘要寫 "prevents overflow/underflow by preserving the IEEE 754 double-precision exponent"，但用途是減少記憶體與通訊量（PageRank）。我們的新意應寫成：在**計算端**（FP64 DGEMM 運算元）截斷，用來降低切換功耗，並在功耗上限下換成時脈。
4. ISC 2025（Ren et al.）已在 **GH200** 上做四精度（FP64/FP32/FP16/FP8）地統計 Cholesky，比只用 FP64 快 3 倍。它寫 "releasing trailing digits of the mantissa of low norm tiles"，審稿人可能以為和我們一樣；實際做法是轉成較低精度的格式。論文要明講這個差別，也不要把我們的 +5.5% 拿去和它的 3 倍比較。
5. 我們「純 FP32 不可用」的結果與前人一致，可以引用佐證。TPDS22 寫 "uniform single-precision is by itself insufficient"；HiPC19 寫對角附近改用單精度時 "the covariance matrix may lose the numerical property of positive definiteness"。
6. 我們的位置是隨機產生的，沒有做 Morton 排序（`apps/geostat/gp_mle.py`：`xy = cp.asarray(rs.random((N, 2)))`）。KAUST 的 band 法與範數法，都要靠排序把強相關集中到對角線附近（TPDS22："thanks to Morton ordering"）。在我們的設定下，tile 法能降精度的 tile 應該很少（我方推論，未實測）。論文要說明這一點，或補一組排序後的對照。
7. cuBLAS 的動態尾數控制，在所需位元超過上限時會 "dynamically dispatches to native FP64"（13.2 文件的預設上限是 79 位）。如果論文報告 `emud`（動態）的結果，要先確認它沒有悄悄退回原生 FP64。
8. 找不到描述 GPU4PySCF DF-MP2 實作的論文（見 §5）。另外，Wu et al. 2024 已正式出版（WIREs CMS 2025），應改引正式版。

## 1. ExaGeoStat 軟體論文 [Abdulah2018]

**書目**：Sameh Abdulah, Hatem Ltaief, Ying Sun, Marc G. Genton, David E. Keyes. *ExaGeoStat: A High Performance Unified Software for Geostatistics on Manycore Systems.* IEEE Transactions on Parallel and Distributed Systems, 29(12):2771–2784, 2018. doi:10.1109/TPDS.2018.2850749。預印本 arXiv:1708.02835（v3，2018-06-22）。

**內容**：以 Matérn 共變異的高斯對數概似做最大概似估計（MLE），"operates on a large dense covariance matrix"。計算是 Chameleon 稠密線性代數庫加 StarPU 執行期上的 tile Cholesky，平台為 "Intel KNLs, NVIDIA GPUs, and distributed-memory systems"。全程 FP64，定位為 "a reference evaluation of statistical parameters, with which to assess the validity of the various approaches based on approximation"。

**查證狀態**：已核對原始頁面（arXiv v3 全文）。

## 2. 同團隊的混合精度地統計研究

### 2a. HiPC 2019 [Abdulah2019]

**書目**：Sameh Abdulah, Hatem Ltaief, Ying Sun, Marc G. Genton, David E. Keyes. *Geostatistical Modeling and Prediction Using
Mixed Precision Tile Cholesky Factorization.* 2019 IEEE 26th International Conference on High Performance Computing, Data, and
Analytics (HiPC), pp. 152–162, 2019. doi:10.1109/HiPC.2019.00028。arXiv:2003.05324。

- **精度與選法**：只用 FP64 與 FP32，依對角帶寬（參數 diag_thick）決定，記為 DP(x%)-SP(y%)；對角 tile 一律 DP。原文：
  "Under an appropriate ordering, it operates with double-precision arithmetic on tiles around the diagonal, while reducing to
  single-precision arithmetic for tiles sufficiently far off."
- **準確度指標**：參數估計（變異數、範圍、平滑度）與預測均方誤差 PMSE。弱、中、強相關（θ2 = 0.03、0.10、0.30）各用 100 組
  40K 合成資料，相關越強損失越多。原文也提到高相關資料的迭代數會增加，有時 "the total execution time of the MLE operation using the
  mixed-precision method exceeded the total time required by the DP arithmetic method"。
- **加速**（DP(10%)-SP(90%) 相對 DP）：摘要寫 "an average of 1.6X"。分平台：Haswell 36 核 1.71X、Skylake 56 核 1.84X；
  K80 1.74X、P100 2.18X、V100 1.82X；Shaheen-II（Cray XC40）64、128、256、512 節點依序為 1.61X、1.45X、1.48X、1.27X。

**查證狀態**：已核對原始頁面（IEEE 出版版 PDF，放在 Genton 的個人網站；數字與 arXiv 版相同）。

### 2b. IEEE TPDS 2022 [Abdulah2022]

**書目**：Sameh Abdulah, Qinglei Cao, Yu Pei, George Bosilca, Jack Dongarra, Marc G. Genton, David E. Keyes, Hatem Ltaief, Ying Sun.
*Accelerating Geostatistical Modeling and Prediction With Mixed-Precision Computations: A High-Productivity Approach With PaRSEC.*
IEEE Transactions on Parallel and Distributed Systems, 33(4):964–976, 2022. doi:10.1109/TPDS.2021.3084071。

- **精度與選法**：FP64/FP32/FP16，仍以 band 決定："we reduce the precision of weakly correlated locations to single- or half-
  precision based on distance"。參數 band_size_dp、band_size_sp，記為 aD:bS:cH。對角 tile 做 DPOTRF、DSYRK（FP64）；
  GEMM 依 m−n 選 DGEMM、SGEMM 或 HGEMM。HP 只用在 GEMM，HP 區的資料以 SP 存放。相關集中在對角線附近要靠 Morton 排序。
- **摘要的數字**："application-expected accuracy worthy of double-precision from a majority half-precision computation, in a context
  where uniform single-precision is by itself insufficient"；"up to 1.59X by mixing FP64/FP32 operations on 1536 nodes of HAWK or
  4096 nodes of Shaheen II, and up to 2.64X by mixing FP64/FP32/FP16 operations on 128 nodes of Summit"（Summit 為 V100），
  分別對應 4.5、4.7、9.1 PFlop/s。
- **準確度**：參數、對數概似（llh）、MSPE、預測不確定度、迭代數。土壤濕度資料：100D 的 llh −59740.65974，1D:99H 為
  −59867.53239，參數也明顯偏離。平滑的風速資料從 10D:90S 起就偏：llh 241480.9994（100D）對 239908.1004（10D:90S）。
  2.64X 是 1D:99H 的結果（"up to 9.1 PFlop/s for 1D:99H ... 2.64X speedup"），也就是土壤濕度資料上參數偏離最多的設定，不過 MSPE 仍接近（我方判讀）。

**查證狀態**：已核對原始頁面（出版版 PDF，Genton 網站；另核對 netlib 上的預印本）。

### 2c. SC22 Gordon Bell 決選 [Cao2022]

**書目**：Qinglei Cao, Sameh Abdulah, Rabab Alomairy, Yu Pei, Pratik Nag, George Bosilca, Jack Dongarra, Marc G. Genton,
David E. Keyes, Hatem Ltaief, Ying Sun. *Reshaping Geostatistical Modeling and Prediction for Extreme-Scale Environmental
Applications.* SC22, pp. 1–12, IEEE, 2022. doi:10.1109/SC41404.2022.00007。KAUST 新聞稿確認是 2022 Gordon Bell Prize
五組決選之一，未得獎。

- **精度與選法**：FP64/FP32/FP16 混合精度，加上 tile low-rank（TLR）。每個 tile 的精度依 Frobenius 範數比決定：
  "a tile Aij with Frobenius norm less than uhigh∗‖A‖F/(NT∗ulow) may be stored in the lower precision"（引自 Higham & Mary 2022）。
  dense 或 TLR 依秩與效能模型決定。演算法中，對角 tile 做 DPOTRF、DSYRK。
- **硬體**：沒有 GPU。準確度實驗在 Shaheen II（Haswell）上做，FP16 用軟體模擬："we trim the operands of the GEMM kernel to FP16
  and call an SGEMM BLAS routine to accumulate in FP32"。效能實驗在 Fugaku（A64FX）上做："We omit the FP16 HGEMM since our MLE
  application requires HGEMM with FP32 accumulation"，所以改用 SGEMM。
- **準確度**：參數、llh、MSPE。土壤濕度資料的 llh：Dense FP64 −52185.7336、MP+dense −52185.7643、MP+dense/TLR −52188.2341。
- **加速**："Performance speedup up to 12X over FP64 execution"。這是 MP+dense/TLR 在 16K 個 Fugaku 節點、弱相關下的結果，
  而且前提是 "If we assume we have enough memory capacity to host a matrix of size 9M in pure FP64"。最大規模為 48,384 節點。
  12X 主要來自 TLR，不只是混合精度（我方判讀）。

**查證狀態**：已核對原始頁面（Genton 網站 PDF；決選身分見 https://www.kaust.edu.sa/en/news/kaust-supercomputing-expertise-shines-at-sc22）。

### 2d. 2023–2025 的 GPU、FP16、FP8 後續研究

**[Cao2023] CLUSTER 2023（有量功耗，最需要注意）**。Qinglei Cao, Sameh Abdulah, Hatem Ltaief, Marc G. Genton, David Keyes,
George Bosilca. *Reducing Data Motion and Energy Consumption of Geospatial Modeling Applications Using Automated Precision
Conversion.* 2023 IEEE International Conference on Cluster Computing (CLUSTER), pp. 330–342, 2023. doi:10.1109/CLUSTER52292.2023.00035。
- **精度與選法**：FP64、FP32、FP16_32（A、B 為 FP16，C 與運算為 FP32）、FP16。對角 tile "they invariably make use of FP64
  precision"；非對角 tile 依 ‖Aij‖×NT/‖A‖ ≤ ureq/ulow 決定。Matérn 的門檻最嚴："only the accuracy of 10−9 can meet the required level of precision"。
- **加速**：摘要寫 "up to 3X"；正文是 64 個 Summit 節點（384 張 V100）上 "up to 3.2X speedup compared to FP64"。
  另有極端設定（對角 FP64、其餘 tile 全部 FP16，不是依準確度門檻選出的）：單 GPU 在 V100 與 A100 超過 11X，H100 PCIe 為 4.7X。
- **功耗**：圖 10 比較單張 V100、A100、H100 上 FP64 與混合精度的功耗、總能耗（J）與 Gflops/W。量法是
  "energy measurements taken at regular intervals using provided Nvidia tools"，並提到 "the real-time power consumption on H100
  does not reach the max TDP"。**沒有設功耗上限，也沒有分析時脈。** 能耗數值只畫在圖中，正文沒有；本輪無法從 PDF 讀出（UNVERIFIED）。
- 查證狀態：已核對原始頁面（出版版 PDF，Genton 網站）。

**[Abdulah2024] SC24 氣候模擬器（2024 ACM Gordon Bell Prize for Climate Modelling 得主，見 OLCF 2024-11-22 新聞）**。
Sameh Abdulah, Allison H. Baker, George Bosilca, Qinglei Cao, Stefano Castruccio, Marc G. Genton, David E. Keyes, Zubair Khalid,
Hatem Ltaief, Yan Song, Georgiy L. Stenchikov, Ying Sun. *Boosting Earth System Model Outputs And Saving PetaBytes in Their
Storage Using Exascale Climate Emulators.* SC24, pp. 1–12, IEEE, 2024. doi:10.1109/SC41406.2024.00008。arXiv:2408.04440。
- **精度與選法**：對球諧係數之 VAR 殘差的經驗共變異矩陣做 Cholesky，採 band 法，四種設定："full double-precision (DP); a single band as DP with
  the remaining tiles as SP (DP/SP); DP with 5% as SP and the remaining tiles as HP (DP/SP/HP); and a single band as DP with the remaining tiles as HP (DP/HP)"。
- **加速**：2,048 個 Summit 節點上相對 DP："speedups of 2X, 3.2X, and 5.2X for DP, DP/SP/HP, and DP/HP"。第一項原文寫 DP，
  依圖 6 應為 DP/SP（我方判讀）。Alps（GH200）1,024 節點上 DP/HP 為 384.2 PFlop/s，每 GPU 93.8 TFlop/s。
- **準確度與功耗**：準確度以模擬場和 ERA5／模式輸出的統計一致性呈現，不是 llh。本文沒有量功耗，只轉述 [Cao2023] 的
  "reduced power consumption"。查證狀態：已核對原始頁面（出版版與 arXiv）。

**[Ren2025] ISC 2025（GH200 上的四精度地統計 Cholesky）**。Jie Ren, Hatem Ltaief, Sameh Abdulah, David E. Keyes. *Accelerating
Mixed-Precision Out-of-Core Cholesky Factorization with Static Task Scheduling.* ISC High Performance 2025 Research Paper
Proceedings (40th International Conference), pp. 1–12, 2025. doi:10.23919/ISC.2025.11017733。arXiv:2410.09819。
- **精度與選法**："a new four-precision (FP64/FP32/FP16/FP8) left-looking Cholesky factorization to take advantage of the application
  resiliency with respect to releasing trailing digits of the mantissa of low norm tiles"。實作是依 Higham–Mary 範數規則把 tile 轉成較低精度格式。
- **加速**："20% performance superiority against cuSOLVER on a single GH200 with FP64"；"a 3X performance speedup against its
  FP64-only counterpart"。單張 GH200 最高 136.1 TFlop/s（最低準確度 1e−5）。
- **準確度與功耗**：準確度以 MxP 與 FP64 概似分佈之間的 KL divergence 衡量。沒有量功耗。
- 查證狀態：內容核對的是 arXiv v1（2024-10-13）全文；ISC 出版版有沒有改動未核對（UNVERIFIED）。

**相鄰研究（GP 式核矩陣，不是地統計）**
- [Ltaief2024] SC24 GWAS：核嶺迴歸（KRR）用 FP64/FP32/FP16/FP8 四精度 Cholesky，"at 1.805 mixed-precision ExaOp/s on a nearly
  full Alps system"（GH200）；距離計算用 INT8 tensor core。doi:10.1109/SC41406.2024.00012。已核對 arXiv:2409.01712 全文。
- [Maddox2022] UAI 2022：GP 訓練改用半精度共軛梯度（CG），因為 "direct methods such as the standard Cholesky decomposition fails
  in half precision due to round-off error"。PMLR 180:1306–1316。已核對 PMLR 全文。
- [Carrica2026] arXiv:2601.08082（只有預印本）：在 H200 與 MI300X 上做遞迴混合精度的 SPD 求解，"assigns low-precision FP16
  arithmetic to large off-diagonal blocks, while preserving high precision on diagonal blocks"，"up to a 5.07x speedup"。只讀了 arXiv 摘要。

**總表**

| 文獻 | 精度 | 如何選 | 對角 tile | 準確度指標 | 硬體 | 量功耗 |
|---|---|---|---|---|---|---|
| HiPC19 | FP64/32 | band | FP64 | 參數、PMSE | Intel CPU、K80/P100/V100、XC40 | 否 |
| TPDS22 | FP64/32/16 | band | FP64 | 參數、llh、MSPE、不確定度 | XC40、HAWK、Summit（V100） | 否 |
| SC22 | FP64/32/16＋TLR | tile 範數 | FP64 | 參數、llh、MSPE | Shaheen II、Fugaku（無 GPU） | 否 |
| CLUSTER23 | FP64/32/16_32/16 | tile 範數 | FP64 | 參數估計 | V100、A100、H100、Summit | **是（單 GPU、無上限）** |
| SC24 | DP/SP/HP | band | FP64 | 統計一致性 | Frontier、Alps（GH200）、Leonardo、Summit | 否 |
| ISC25 | FP64/32/16/8 | tile 範數 | 原文未明講 | KL divergence | GH200、A100、H100 | 否 |

## 3. 新穎性核對：FP64 內截斷尾數、GPU 功耗與功耗上限

- **地統計與 GP 文獻**：上表沒有一篇在 FP64 格式內截斷尾數、保留 11 位指數；全部是換格式，收益來自較快的低精度單元與較少的資料搬移。
  形式上最接近的是 SC22 在 CPU 上 "trim the operands of the GEMM kernel to FP16" 後用 SGEMM 累加。那是為了在沒有 FP16 GEMM 的
  硬體上模擬 FP16 的數值；指數範圍有沒有保留，原文沒寫（UNVERIFIED），而且與功耗無關。
- **功耗**：只有 [Cao2023] 實測過，條件是單張 GPU、沒有功耗上限，結論是低精度能降低能耗。沒有一篇設功耗上限或討論時脈。
- **地統計以外**：CPMS 已經在 FP64 內「丟尾數、留指數」，但目的是減少記憶體與通訊量。[Grutzmacher2020] 摘要：
  "prevents overflow/underflow by preserving the IEEE 754 double-precision exponent"、"the truncated values (omitting significand bits)"；
  在 V100 與 Xeon 上，PageRank "about 10% faster if high-accuracy output is needed, and about 30% faster if reduced output accuracy is acceptable"。
  同團隊另有 CCPE 2020 的格式論文（doi:10.1002/cpe.5418）。硬體上以尾數歸零省電的前例（Tong2000、AMD 專利等）見 `round4_refs.md`。
- **建議的主張寫法**：「據我們所知，這是第一個在 GP 概似的 FP64 Cholesky 中保留 FP64 格式、只截斷 DGEMM 運算元的尾數，
  並在 GPU 模組功耗上限下量測速度與能耗的研究。」同時明確引用 [Cao2023]（量過功耗）與 [Grutzmacher2020]（FP64 內截斷，目的是頻寬）。
- 查證狀態：CPMS 兩篇的書目經 Crossref 核對，內容只讀到摘要（OpenAlex 轉載的出版社摘要）。

## 4. cuBLAS FP64 模擬（fixed-point／Ozaki）

- **官方文件**：NVIDIA cuBLAS 文件 §1.5 "Floating Point Emulation"。目前版本是 v13.4（頁面 "Last updated on Sep 13, 2026"）：
  https://docs.nvidia.com/cuda/cublas/index.html#floating-point-emulation 。另核對了與我們工具鏈（CUDA 13.2）對應的存檔版
  13.2.1（"Last updated on Apr 02, 2026"）：https://docs.nvidia.com/cuda/archive/13.2.1/cublas/index.html 。
- **動態尾數控制一句話**：列舉型別 `cudaEmulationMantissaControl_t` 的說明，在 13.0.2、13.2.x、13.4 三個版本中文字相同：
  "CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC: The number of retained mantissa bits is computed at runtime to ensure the same or better
  accuracy than the native floating point representation."
  §1.5.2.1（v13.4）補充："Our automatic dynamic precision (ADP) framework computes the proper number of fixed-point mantissa bits
  required to maintain accuracy equal or better than FP64. If the number of required mantissa bits exceeds a library-defined default
  ... or a user-provided maximum number of bits ..., the framework dynamically dispatches to native FP64."
  13.2.1 的預設上限：動態 79 位、固定 55 位。
- **啟用方式**：`cublasSetMathMode(handle, CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH)`（"Enable acceleration of double-precision routines
  using fixed-point emulation algorithms"），或使用 compute type `CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT`。尾數模式用
  `cublasSetFixedPointEmulationMantissaControl()` 設定，位元數用 `cublasSetFixedPointEmulationMaxMantissaBitCount()`。
  FP64 模擬支援 compute capability 8.x、9.0（含 GH200）、10.x、11.0、12.x，需要 CUDA 13.0u2 以上。
- **與「列內指數跨度」論點直接相關的原文**：13.2.1 寫 "The scaling factor is shared for elements in the same row of the A matrix
  or column of the B matrix and is used to logically scale all elements to be between -1 and 1 inclusively."；v13.4 寫
  "so the dynamic range within an A row or B column is encoded as part of the mantissa"。切片數為
  "sliceCount = ceildiv(mantissaBitCount + 1, 8)"，且 "the computational cost grows quadratically with the number of slices"。
  換算：36 位為 5 片，55 位為 7 片（我方計算）。
- **技術部落格**：Cole Brower, John Gunnels, Graham Lopez. *Unlocking Tensor Core Performance with Floating Point Emulation in cuBLAS.*
  NVIDIA Technical Blog，2025-10-24（2025-11-13 更新）。https://developer.nvidia.com/blog/unlocking-tensor-core-performance-with-floating-point-emulation-in-cublas/
  關鍵句："because the Ozaki Scheme uses a fixed-point representation for the operands after their exponents are aligned, the number of
  “mantissa bits” required is data dependent and must be greater than or equal to the 53 bits in the IEEE 754 FP64 representation to
  deliver the same or better accuracy." 注意：文中的效能示範都在 Blackwell（GB200 NVL72、RTX PRO 6000）上，不是 Hopper。
- **ADP 演算法論文**（文件內的連結）：[Schwarz2026]，arXiv:2511.13778。摘要："At its core is the Exponent Span Capacity (ESC), a
  hardware-agnostic estimator that conservatively determines the decomposition parameter (also known as slices) required to achieve
  FP64-level accuracy"；額外負擔 "less than 10% run time overhead"；55 位時，在 GB200 與 RTX Pro 6000 上分別比原生 FP64 GEMM
  快 "up to 2.3x and 13.2x"。ADP 是依指數跨度決定切片數，正好對應我們觀察到「面板 88% 的元素比所在列的最大值小 2^-32 以上」。
- 查證狀態：已核對原始頁面（文件 v13.4 與 13.2.1、部落格頁面、arXiv 摘要；SCA 的書目經 Crossref）。

## 5. GPU4PySCF DF-MP2

- **結論：找不到描述 GPU4PySCF DF-MP2 實作的論文。** 依據如下：
  (1) Wu et al.（arXiv:2404.09452）已正式出版於 WIREs Comput. Mol. Sci.，標題改為複數 "Frameworks"。表 4 的 MP2 列只在第一欄（SCF，即能量）標
  "GPU"，定義是 "drop-in replacement or naïve implementation"；梯度與 Hessian 為 "CPU"。正文寫
  "Our work in GPU4PySCF includes preliminary implementations for MP2 and CCSD methods"。
  (2) Li et al.（arXiv:2407.09700，已出版於 J. Phys. Chem. A）正文沒有 MP2。
  (3) PySCF 十年回顧（arXiv:2603.14155v2）寫 "For post-HF methods, GPU4PySCF (as of version 1.4.2) only supports the integral-in-core
  algorithms for restricted MP2 and CCSD."
  (4) 用 arXiv API 搜尋 "gpu4pyscf"（9 篇）與 "MP2 AND GPU"，都沒有相關論文。
- **程式來源**：我們用的 v1.8.1 中，`gpu4pyscf/mp/dfmp2.py` 來自 GitHub PR #662 "Density-Fitting MP2 (single-GPU)"
  （作者 ajz34，2026-02-24 合併；CHANGELOG v1.6.0（2026-03-10）記為 "Density-Fitting MP2 (#662)"）。PR 說明寫：
  "This implementation supports FP32 in MP2 O(N^5) contraction, which is the largest efficiency bottleneck. Usually FP32 is sufficiently acccurate."
  README 把 "MP2/DF-MP2 and CCSD" 列在 "still in the experimental stage"，References 段落列的是 Li 2024 與 Wu 2024。
- **建議**：GPU4PySCF 的一般性引用改用正式版 [Wu2025]，並加上 [Li2025]；DF-MP2 程式用軟體引用（GitHub、v1.8.1、PR #662）。
  若要提 RI-MP2 降精度的前例，可引 [OlivaresAmaya2010]：在 GPU 上用 SGEMM 與混合精度 GEMM 做 RI-MP2；纈胺黴素（168 原子、
  cc-pVDZ）的相關能誤差 "reduced from -10.0 to -1.2 kcal mol(-1) for SGEMM and MGEMM, respectively"（PubMed 摘要）。
- 查證狀態：GitHub PR 與 CHANGELOG 已核對。[Wu2025] 書目經 Crossref（Wiley 頁面擋自動讀取）。[OlivaresAmaya2010] 書目經
  Crossref，內容只讀到摘要。

## 可直接貼進論文的書目清單

```
[Abdulah2018] Sameh Abdulah, Hatem Ltaief, Ying Sun, Marc G. Genton, David E. Keyes. ExaGeoStat: A High Performance Unified Software for Geostatistics on Manycore Systems. IEEE Transactions on Parallel and Distributed Systems, 29(12):2771–2784, 2018. doi:10.1109/TPDS.2018.2850749
[Abdulah2019] Sameh Abdulah, Hatem Ltaief, Ying Sun, Marc G. Genton, David E. Keyes. Geostatistical Modeling and Prediction Using Mixed Precision Tile Cholesky Factorization. In 2019 IEEE 26th International Conference on High Performance Computing, Data, and Analytics (HiPC), pp. 152–162, 2019. doi:10.1109/HiPC.2019.00028
[Abdulah2022] Sameh Abdulah, Qinglei Cao, Yu Pei, George Bosilca, Jack Dongarra, Marc G. Genton, David E. Keyes, Hatem Ltaief, Ying Sun. Accelerating Geostatistical Modeling and Prediction With Mixed-Precision Computations: A High-Productivity Approach With PaRSEC. IEEE Transactions on Parallel and Distributed Systems, 33(4):964–976, 2022. doi:10.1109/TPDS.2021.3084071
[Cao2022] Qinglei Cao, Sameh Abdulah, Rabab Alomairy, Yu Pei, Pratik Nag, George Bosilca, Jack Dongarra, Marc G. Genton, David E. Keyes, Hatem Ltaief, Ying Sun. Reshaping Geostatistical Modeling and Prediction for Extreme-Scale Environmental Applications. In SC22: International Conference for High Performance Computing, Networking, Storage and Analysis, pp. 1–12, 2022. doi:10.1109/SC41404.2022.00007
[Cao2023] Qinglei Cao, Sameh Abdulah, Hatem Ltaief, Marc G. Genton, David Keyes, George Bosilca. Reducing Data Motion and Energy Consumption of Geospatial Modeling Applications Using Automated Precision Conversion. In 2023 IEEE International Conference on Cluster Computing (CLUSTER), pp. 330–342, 2023. doi:10.1109/CLUSTER52292.2023.00035
[Abdulah2024] Sameh Abdulah, Allison H. Baker, George Bosilca, Qinglei Cao, Stefano Castruccio, Marc G. Genton, David E. Keyes, Zubair Khalid, Hatem Ltaief, Yan Song, Georgiy L. Stenchikov, Ying Sun. Boosting Earth System Model Outputs And Saving PetaBytes in Their Storage Using Exascale Climate Emulators. In SC24: International Conference for High Performance Computing, Networking, Storage and Analysis, pp. 1–12, 2024. doi:10.1109/SC41406.2024.00008
[Ren2025] Jie Ren, Hatem Ltaief, Sameh Abdulah, David E. Keyes. Accelerating Mixed-Precision Out-of-Core Cholesky Factorization with Static Task Scheduling. In ISC High Performance 2025 Research Paper Proceedings (40th International Conference), pp. 1–12, 2025. doi:10.23919/ISC.2025.11017733
[Ltaief2024] Hatem Ltaief, Rabab Alomairy, Qinglei Cao, Jie Ren, Lotfi Slim, Thorsten Kurth, Benedikt Dorschner, Salim Bougouffa, Rached Abdelkhalak, David E. Keyes. Toward Capturing Genetic Epistasis From Multivariate Genome-Wide Association Studies Using Mixed-Precision Kernel Ridge Regression. In SC24: International Conference for High Performance Computing, Networking, Storage and Analysis, pp. 1–12, 2024. doi:10.1109/SC41406.2024.00012
[Maddox2022] Wesley J. Maddox, Andres Potapczynski, Andrew Gordon Wilson. Low-Precision Arithmetic for Fast Gaussian Processes. In Proceedings of the 38th Conference on Uncertainty in Artificial Intelligence (UAI), PMLR 180:1306–1316, 2022.
[Carrica2026] Vicki Carrica, Rabab Alomairy, Evelyne Ringoot, Alan Edelman. Hierarchical Recursive Precision for Accelerating Symmetric Linear Solves on MXUs. arXiv preprint arXiv:2601.08082, 2026.
[HighamMary2022] Nicholas J. Higham, Theo Mary. Mixed precision algorithms in numerical linear algebra. Acta Numerica, 31:347–414, 2022. doi:10.1017/S0962492922000022
[Grutzmacher2020] Thomas Grützmacher, Terry Cojean, Goran Flegar, Hartwig Anzt, Enrique S. Quintana-Ortí. Acceleration of PageRank with Customized Precision Based on Mantissa Segmentation. ACM Transactions on Parallel Computing, 7(1):1–19, 2020. doi:10.1145/3380934
[cuBLASEmu] NVIDIA. cuBLAS Library Documentation, Section 1.5 "Floating Point Emulation" (v13.4; CUDA 13.2.1 archived version also consulted), accessed Oct. 5, 2026. https://docs.nvidia.com/cuda/cublas/index.html#floating-point-emulation
[Brower2025] Cole Brower, John Gunnels, Graham Lopez. Unlocking Tensor Core Performance with Floating Point Emulation in cuBLAS. NVIDIA Technical Blog, Oct. 24, 2025. https://developer.nvidia.com/blog/unlocking-tensor-core-performance-with-floating-point-emulation-in-cublas/
[Schwarz2026] Angelika Schwarz, Anton Anders, Cole Brower, Harun Bayraktar, John Gunnels, Kate Clark, RuQing G. Xu, Samuel Rodriguez, Sebastien Cayrols, Paweł Tabaszewski, Victor Podlozhnyuk. Guaranteed DGEMM Accuracy While Using Reduced Precision Tensor Cores Through Extensions of the Ozaki Scheme. In Proceedings of the Supercomputing Asia and International Conference on High Performance Computing in Asia Pacific Region (SCA/HPCAsia 2026), pp. 91–101, 2026. doi:10.1145/3773656.3773670
[Wu2025] Xiaojie Wu, Qiming Sun, Zhichen Pu, Tianze Zheng, Wenzhi Ma, Wen Yan, Yu Xia, Zhengxiao Wu, Mian Huo, Xiang Li, Weiluo Ren, Sheng Gong, Yumin Zhang, Weihao Gao. Enhancing GPU-Acceleration in the Python-Based Simulations of Chemistry Frameworks. WIREs Computational Molecular Science, 15(2):e70008, 2025. doi:10.1002/wcms.70008
[Li2025] Rui Li, Qiming Sun, Xing Zhang, Garnet Kin-Lic Chan. Introducing GPU Acceleration into the Python-Based Simulations of Chemistry Framework. The Journal of Physical Chemistry A, 129(5):1459–1468, 2025. doi:10.1021/acs.jpca.4c05876
[GPU4PySCF-DFMP2] GPU4PySCF developers. gpu4pyscf v1.8.1, module gpu4pyscf/mp/dfmp2.py (density-fitting MP2 added in pull request #662, "Density-Fitting MP2 (single-GPU)", merged Feb. 24, 2026). https://github.com/pyscf/gpu4pyscf
[OlivaresAmaya2010] Roberto Olivares-Amaya, Mark A. Watson, Richard G. Edgar, Leslie Vogt, Yihan Shao, Alán Aspuru-Guzik. Accelerating Correlated Quantum Chemistry Calculations Using Graphical Processing Units and a Mixed Precision Matrix Multiplication Library. Journal of Chemical Theory and Computation, 6(1):135–144, 2010. doi:10.1021/ct900543q
```

## 論文定位建議

ExaGeoStat 系列 [Abdulah2019, Abdulah2022, Cao2022, Cao2023, Abdulah2024, Ren2025] 已經證明，GP 概似的 Cholesky 可以大量改用低精度：他們靠空間排序把強相關集中到對角線附近，對角 tile 留在 FP64，遠離對角或範數小的整個 tile 換成 FP32、FP16 或 FP8；收益來自 tensor core 等較快的低精度單元與較少的資料搬移，在 GPU 上可達數倍加速，[Cao2023] 也量到能耗下降。我們的做法有三點不同，而且可以和他們的方法疊加。
第一，不換格式：每個元素保留自己的 11 位 FP64 指數，只把 DGEMM 運算元的尾數截到 28 位（仍多於 FP32 的 23 位），運算單元、資料量與程式路徑都不變。因此元素之間的指數跨度不影響截斷，也不需要依範數或排序挑 tile。面板中 88% 的元素比所在列的最大值小 2^-32 以上；對每列共用一個縮放因子的定點模擬（cuBLAS Ozaki，切片數隨所需尾數位元增加 [cuBLASEmu, Schwarz2026]），這是額外成本。
第二，收益機制不同：翻轉的位元減少，功耗下降，在模組功耗上限下時脈因此上升。所以只有在功耗受限時才有約 +5.5%；1000 W 下速度幾乎不變，能耗降 9–10%。這個幅度遠小於換格式的方法，論文不應直接比較倍數。
第三，兩者互補：tile 法中仍留在 FP64 的 tile 可以再套用截斷。強相關或門檻嚴格時，這類 tile 更多（[Cao2023] 中 Matérn 需要 1e−9 的門檻，3D 平方指數核有超過 60% 的 tile 用 FP64 或 FP32），可套用截斷的空間也更大（我方推論）。
另外要寫明兩個限制：位置沒有做 Morton 排序，tile 法在此設定下難以發揮（推論，未實測）；核心是自己寫的、用合成資料。純 FP32 不可用的結果，可引用 [Abdulah2019]、[Abdulah2022] 的觀察佐證。

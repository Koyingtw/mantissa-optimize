# 相關研究調查：精度調校、功耗上限、混合精度儲存、資料相依的資料搬移能耗

查證日期：2026-09-29。每一筆都開過原始頁面：arXiv 摘要頁、出版社或作者放的 PDF（已抽出全文核對）、
Crossref DOI 紀錄（`api.crossref.org/works/<DOI>`）、Semantic Scholar 紀錄、ORNL／大學的出版品頁面，
或官方規則頁。只看到搜尋摘要、沒有打開原始頁面的項目，都標為「未查證」。
本檔補充 `references.md`（Gregersen2024、Bhalachandra2022、Haidar2018、Klöwer2021 等已收錄的文獻），
不取代它。

本研究的定位（寫相關研究時用）：**同一格式、同樣的 byte 數與指令，只把運算元的低位尾數清為 0**，
讓翻轉的位元變少、功耗下降。在模組功耗上限下，時脈因此升高，速度提升 3–28%；不設上限時，能耗降低 7–18%。
測試範圍涵蓋 FP64/BF16/FP8 GEMM、FP64 CUDA-core FMA、記憶體複製、HPL、HPL-MxP、QE、PySCF，以及 LLM 推論與訓練。

---

## 結論摘要（建議優先引用的 10 組文獻）

1. **Gregersen, Patel, Choukse — Input-Dependent Power Usage in GPUs（SC24-W）**：已證實 GEMM 功耗會隨輸入資料變動，差距可達約 40%，也觀察到清掉最低位元後功耗下降。
   我們的差異：把這個現象當成手段，在功耗上限下換成時脈與速度，並把範圍擴大到 FP64、FP8、記憶體複製與完整應用，還附上數值正確性檢查。
2. **Bhalachandra et al. — Understanding the Impact of Input Entropy on FPU, CPU, and GPU Power（arXiv 2022）**：A100 上的 DGEMM 在相同尺寸下，只換數值，功耗就差到 67%。
   他們的結論是功耗預算與 benchmark 方法應該考慮輸入熵；我們則是主動降低熵來換取效能或能耗。
3. **AMD 專利 US11455766B2「Variable Precision Computing System」（2022 核准）**：GPU shader 中加入硬體 zeroing module，
   把運算元尾數的一部分設為 0 來省電，應用在圖形渲染。**這是最接近的先前技術，論文不能宣稱「首次以尾數歸零省 GPU 功耗」。**
   我們的差異：純軟體做法、硬體不改（NVIDIA GH200）、針對 HPC 數值（FP64、HPL 殘差檢查），並量化「功耗上限→時脈→速度」這條機制。
4. **Tong, Nagle, Rutenbar — Reducing power by optimizing the necessary precision/range of FP arithmetic（IEEE TVLSI 2000）**：
   尾數位寬砍半時，多數程式的輸出不變；可變位寬 FPU 可讓乘法器每次運算的能量最多降 66%。這是在硬體面改位寬；我們在現成 GPU 上保留原格式，只改資料值。
5. **Precimonious（SC13）及其後續（Blame Analysis、HiFPTuner、ADAPT、GPUMixer、AMPT-GA）**：這些工具搜尋變數或指令的*型別*，
   例如把 double 改成 float，目標主要是速度與精度，能耗通常只在動機裡提到。我們不換型別，也就不需要低精度單元或型別轉換；
   收益來自功耗與時脈。這類 tuner 可以反過來用來決定每個變數要清幾個位元，兩者互補。
6. **rpe（Dawson & Düben, GMD 2017）與 chop（Higham & Pranesh, SISC 2019）**：兩者都在*同一個高精度格式*內把尾數低位清 0
   （rpe 用 round-to-nearest-even），但目的是*模擬*低精度來研究誤差，會增加額外開銷，也不量功耗。
   我們用的是同一種位元操作，但拿來當功耗旋鈕。
7. **Ginkgo memory accessor／compressed-basis GMRES（Aliaga, Anzt, Grützmacher, Quintana-Ortí, Tomás, IJHPCA 2023；
   Grützmacher et al., SPE 2023）與 Higham & Mary（Acta Numerica 2022）**：這些做法把儲存格式和運算格式分開，
   以*減少 byte 數*來突破頻寬瓶頸。我們的做法是 byte 數不變、翻轉位元變少，兩者可以疊加使用。
8. **Chatterjee et al.（HPCA 2017）與 O'Connor et al.（MICRO 2017）**：依 NVIDIA 的 HBM 能量模型，HBM 的 column energy 取決於翻轉率：
   翻轉率 0% 時約 1.5 pJ/bit，100% 時約 5.7 pJ/bit；HBM 採用 DBI-AC。這說明了為什麼記憶體複製的能耗也會隨尾數歸零下降；
   我們則是在真實的 GH200 上做端到端量測。
9. **Ghose et al.（POMACS／SIGMETRICS 2018）**：在實機 DDR3L 上量到 DRAM 電流會隨資料中 1 的個數以及翻轉位元數線性變化。
   這是「記憶體能耗取決於資料值」最直接的實測證據，但測的是 DDR3L，不是 GPU 的 HBM。
10. **功耗上限研究：Patrou et al.（ISC25-W，GH200 功耗上限與 automatic power steering）、Zhao et al.（SoCC 2023）、
    Yang, Adamek, Armour（SC24，NVIDIA 內建功耗感測器）**：前兩者把功耗上限當成在效能與能耗之間取捨的旋鈕，
    我們則在上限固定時降低工作負載本身的功耗需求，與調上限的做法正交、可以併用。
    Yang et al. 指出 nvidia-smi 在 A100/H100 上只取樣約 25% 的執行時間，這是我們量測方法必須正面回應的問題。

---

## 1. 精度調校（precision tuning）

**共同特徵與我們的差異**：精度調校工具搜尋「哪些變數或運算可以改成*較低精度的格式*」，
例如 double→float，或客製的 exponent/significand 寬度。收益來自較快的低精度單元、較少的 byte 數或 SIMD 寬度，
所以必須付出型別轉換的代價，並依賴硬體支援對應格式。能耗大多只出現在動機中，實際量過的只有模擬器
（ISLPED 2023）或專用硬體（transprecision）。我們的做法：(a) 格式、指令與 byte 數都不變，只把尾數的低 k 位清為 0；
(b) 收益機制是切換活動減少→功耗下降→在功耗上限下時脈上升；(c) 以應用自帶的正確性檢查（HPL 殘差、
SCF 收斂等）驗收。這些 tuner 的敏感度分析（例如 ADAPT）可以用來決定每個變數的 k，屬於互補關係。

- **CRAFT — Lam, Hollingsworth, de Supinski, LeGendre, "Automatically adapting programs for mixed-precision floating-point computation", ICS 2013, pp. 369–378**：
  以二進位改寫自動找出可以改用單精度的指令，是精度調校的早期代表。目標是效能，未量測能耗。
  https://doi.org/10.1145/2464996.2465018（Crossref 紀錄已核對）
- **Precimonious — Rubio-González et al., SC13**：以動態分析與 delta-debugging 搜尋*變數型別*，在精度限制下降低精度，
  GSL、NAS 等程式最多加速 41%。引言把能耗列為高精度的代價之一，但實驗只量執行時間。
  https://doi.org/10.1145/2503210.2503296（作者 PDF 與 Crossref 均已核對）
- **Blame Analysis — Rubio-González et al., ICSE 2016**：只執行一次程式，就用多種精度判斷每個運算元需要的精度；
  與 Precimonious 合用時分析快 9 倍（最多 38 倍），程式最多加速 39.9%。未量測能耗。
  https://doi.org/10.1145/2884781.2884850（作者 PDF 已核對）
- **FPTuner — Chiang et al., POPL 2017, pp. 300–315**：以 Symbolic Taylor Expansions 求出嚴格的誤差界，把精度分配寫成二次規劃（QCQP）求解，
  自動插入升／降精度轉換。POPL 頁面的摘要提到可在精度與效率成本之間權衡。
  https://doi.org/10.1145/3009837.3009846（POPL 2017 活動頁與 Crossref 均已核對）
- **HiFPTuner — Guo & Rubio-González, ISSTA 2018**：以依賴圖加上社群偵測建立階層式搜尋，平均少花 59.6% 的搜尋時間。
  https://doi.org/10.1145/3213846.3213862（作者 PDF 已核對）
- **ADAPT — Menon et al., SC18, pp. 614–626**：用 algorithmic differentiation 估計每個變數降精度後的輸出誤差，
  LULESH 加速 1.2 倍。摘要把「reducing power consumption」列為混合精度的動機，但沒有量測功耗。
  https://doi.org/10.1109/SC.2018.00051（作者 PDF 與 Crossref 均已核對）
- **GPUMixer — Laguna, Wood, Singh, Bagchi, ISC 2019（LNCS）**：第一個 GPU 混合精度 tuner，採「效能導向」：
  用靜態分析找出型別轉換最少的 Fast Imprecise Sets，並以 GPU shadow computation 估計誤差，達到理想加速的 46.4%。
  沒有涉及能耗。https://doi.org/10.1007/978-3-030-20656-7_12（作者 camera-ready PDF 與 Semantic Scholar 均已核對）
- **AMPT-GA — Kotipalli, Singh, Wood, Laguna, Bagchi, ICS 2019, pp. 160–170**：針對 GPU 應用，以 casting-aware 效能模型加上遺傳演算法選擇精度，
  比 Precimonious 好 14–63%。摘要沒有提能耗。https://doi.org/10.1145/3330345.3330360（Crossref 與 Semantic Scholar 均已核對）
- **PreScaler — Kang, Choi, Park, CGO 2020, pp. 280–292**：以記憶體物件為單位，在異質系統（GPU）上做精度縮放，平均加速 1.33 倍。
  摘要沒有提能耗。https://doi.org/10.1145/3368826.3377917（Crossref 與 Semantic Scholar 均已核對）
- **Brunie, Iancu, Ibrahim, Brisk, Cook, "Tuning Floating-Point Precision Using Dynamic Program Information and Temporal Locality", SC20**：
  以指令為搜尋空間，結合動態行為與時間局部性，對完整應用做精度調校。
  https://doi.org/10.1109/SC41405.2020.00054（Crossref 已核對）
- **Wang & Rubio-González, "Predicting Performance and Accuracy of Mixed-Precision Programs for Precision Tuning"（FPLearner）, ICSE 2024**：
  用 GNN 預測混合精度程式的效能與精度，讓 Precimonious、HiFPTuner 的搜尋時間平均少 25.54%。
  https://doi.org/10.1145/3597503.3623338（作者 PDF 已核對）
- **RAPTOR — Hoerold, Ivanov, Dhruv, Moses, Dubey, Wahib, Domke, SC25, pp. 661–680**：以 LLVM 對科學應用做數值剖析，
  可以透明地把高精度運算換成低精度，或*模擬使用者指定的精度*，應用在 Flash-X。摘要沒有提能耗。
  https://doi.org/10.1145/3712285.3759810（arXiv:2507.04647 與 Crossref 均已核對）
- **Chen, Hilaire, Jézéquel, "Floating-point autotuning with customized precisions", arXiv:2606.08339（2026）**：以 PROMISE 搜尋
  客製的 exponent/significand 寬度。摘要把能耗列為動機，沒有實測。https://arxiv.org/abs/2606.08339
- **Carson & Chen, "Precision autotuning for linear solvers via contextual bandit-based RL", arXiv:2601.00728（2026）**：
  用強化學習在迭代修正中動態選擇 FP16/FP32/FP64，成本以計算量衡量，不涉及功耗。https://arxiv.org/abs/2601.00728
- **Adjé, Ben Khalifa, Martel, "Fast and Efficient Bit-Level Precision Tuning", arXiv:2103.05241（2021）**：以整數線性規劃與
  policy iteration，對每個值求出*位元層級*所需的有效位數（最小資料型別）。摘要沒有提能耗。https://arxiv.org/abs/2103.05241
  （出版版本為 SAS 2021，此點未查證）
- **以能耗為目標的精度調校（少數）**：
  - **Li, Wu, Zhang, "Multi-Objective Optimization for Floating Point Mix-Precision Tuning", ISLPED 2023**：以 AD 模擬精度，並以
    Gaussian process 建立效能與功耗模型，做多目標 Bayesian optimization。功耗是在 gem5 模擬器上取得，不是實機量測；
    SIMD 情境下平均省電 17%。https://doi.org/10.1109/ISLPED58423.2023.10244539（作者 PDF 與 Crossref 均已核對）
  - **Tagliavini, Mach, Rossi, Marongiu, Benini, "A transprecision floating-point platform for ultra-low power computing", DATE 2018, pp. 1051–1056**：
    新增 binary8 與 binary16alt 格式並提供硬體支援，最多 90% 的 FP 運算可以降到 8 或 16 位元，能耗最多降低 30%。
    這條路線需要新硬體與新格式。https://doi.org/10.23919/DATE.2018.8342167（arXiv:1711.10374 與 Crossref 均已核對）
  - **Tong, Nagle, Rutenbar, IEEE TVLSI 8(3):273–286, 2000**：見「結論摘要」第 4 點。可變位寬的 FP 乘法器最多省 66% 能量，
    是「少用尾數位元→省電」最早的硬體證據之一。https://doi.org/10.1109/92.845894（Illinois Experts 頁面已核對）
- **同格式內歸零尾數的「模擬器」**（與我們使用同一種位元操作）：
  - **Dawson & Düben, "rpe v5: an emulator for reduced floating-point precision in large numerical simulations", GMD 10:2221–2230, 2017**：
    Fortran 的 `rpe_var` 在每次賦值與運算後，把 significand 截到 n 位（多餘的低位設為 0），並採 IEEE round-to-nearest-even
    避免偏差，數值仍存成 double。論文提到「可能減少功耗」，但這只是研究低精度可行性的工具。
    https://doi.org/10.5194/gmd-10-2221-2017（GMD 頁面與 PDF §2.2 均已核對）
  - **Higham & Pranesh, "Simulating Low Precision Floating-Point Arithmetic", SIAM J. Sci. Comput. 41(5):C585–C602, 2019**：
    MATLAB 的 `chop` 在較高精度中模擬 fp16/bfloat16 等格式，支援多種捨入方式與隨機位元翻轉。
    https://doi.org/10.1137/19M1251308（MIMS EPrints 與 Crossref 均已核對）
- **AMD 專利 US11455766B2（Argade, Peroni；2018 申請、2022 核准）**：shader 的 zeroing module 依 precision flag，
  把運算元尾數的一部分設為 0，符號與指數不動；專利文字寫明「因為部分資料已歸零，功能單元處理時耗電較少」。
  屬於專利，非同儕審查論文，但必須引用並明確區分（見「結論摘要」第 3 點）。
  https://patents.google.com/patent/US11455766B2/en（Google Patents 頁面已核對）

## 2. HPC 與 GPU 的功耗上限（power capping）

**共同特徵與我們的差異**：這類研究把功耗上限（或頻率上限）當成*選擇題*：找出能耗最佳或 EDP 最佳的上限值，
在系統層級做功耗預算與排程，或透過超額配置提高系統吞吐量。應用本身的功耗需求被當成固定的，
「效能換能耗」幾乎都是有損的取捨。我們在上限*固定*時，靠資料值讓同一份工作的功耗需求下降，
時脈因此回升，同時拿到更快的速度與更低的能耗；不設上限時則直接省能。這和調整上限的做法正交，可以併用。

- **David, Gorbatov, Hanebutte, Khanna, Le, "RAPL: Memory power estimation and capping", ISLPED 2010, pp. 189–194**：Intel RAPL
  的記憶體功耗估計與上限機制。https://doi.org/10.1145/1840845.1840883（Crossref 已核對）
- **Rountree, Ahn, de Supinski, Lowenthal, Schulz, "Beyond DVFS: A first look at performance under a hardware-enforced power bound", IPDPSW 2012, pp. 947–953**：
  Sandy Bridge 的硬體功耗上限。處理器的製程變異在沒有上限時不影響效能，但加上上限後就轉成效能差異。
  https://doi.org/10.1109/IPDPSW.2012.116（Univ. of Arizona 出版品頁面已核對）
- **Patki, Lowenthal, Rountree, Schulz, de Supinski, "Exploring hardware overprovisioning in power-constrained, high performance computing", ICS 2013, pp. 173–182**：
  以硬體超額配置搭配硬體功耗上限，平均比最壞情況配置快 50% 以上。https://doi.org/10.1145/2464996.2465009（TUM 出版品頁面已核對）
- **Patki et al., "Practical Resource Management in Power-Constrained, High Performance Computing"（RMAP）, HPDC 2015, pp. 121–132**：
  在設施層級的功耗預算下做 job 層級的功耗分配、超額配置與 power-aware backfilling，平均周轉時間最多快 31% 與 54%。
  https://doi.org/10.1145/2749246.2749262（ORNL 出版品頁面已核對）
- **Inadomi et al., "Analyzing and mitigating the impact of manufacturing variability in power-constrained supercomputing", SC15**：
  在 4 套系統上，製程變異於功耗限制下轉成效能不均；依變異分配功耗後，在 1,920 socket 的系統上最多加速 5.4 倍，平均 1.8 倍。
  https://doi.org/10.1145/2807591.2807638（Crossref 與 Semantic Scholar 摘要均已核對）
- **Haidar, Jagode, Vaccaro, YarKhan, Tomov, Dongarra, "Investigating power capping toward energy-efficient scientific applications", CCPE 31(6):e4485, 2019**：
  在 KNL 上研究不同計算強度的數值核心在功耗上限下的效能與能耗。https://doi.org/10.1002/cpe.4485（ORNL 出版品頁面已核對）
- **Krzywaniak & Czarnul, "Performance/Energy Aware Optimization of Parallel Applications on GPUs Under Power Capping", PPAM 2019（LNCS 12044, 2020）, pp. 123–133**：
  用 NVIDIA GPU 的功耗上限跑 NPB BT/SP/LU 與 cuBLAS GEMM，不同程式的最佳上限不同。
  https://doi.org/10.1007/978-3-030-43222-5_11（Crossref 已核對；使用的 GPU 型號未查證）
- **McDonald et al., "Great Power, Great Responsibility: Recommendations for Reducing Energy for Training Language Models", Findings of NAACL 2022, pp. 1962–1970**：
  訓練 transformer 語言模型時，功耗上限可省 15% 能耗，時間只略增。https://doi.org/10.18653/v1/2022.findings-naacl.151（ACL Anthology 已核對）
- **Zhao et al., "Sustainable Supercomputing for AI: GPU Power Capping at HPC Scale", SoCC 2023, pp. 588–596**：MIT 超級電腦
  （主要是 V100，另有少量 A100）全系統套用約 60% 的上限。GPU 平均降約 10 W、4.8 °C；上限 200 W 時訓練能耗降約 10%，速度損失低於 5%；
  LLaMA-65B 推論在 175 W 上限時慢 5–8%，省能 22–24%。https://doi.org/10.1145/3620678.3624793（arXiv:2402.18593 與 Crossref 均已核對）
- **Patel et al., "Characterizing Power Management Opportunities for LLMs in the Cloud"（POLCA）, ASPLOS 2024, pp. 207–222**：
  分析 LLM 訓練與推論的功耗型態，以功耗超額訂閱讓既有叢集多放 30% 的伺服器。https://doi.org/10.1145/3620666.3651329（Microsoft Research 頁面與 Crossref 均已核對）
- **Yoshida, Sageyama, Miwa, Yamaki, Honda, "Analyzing Performance and Power-Efficiency Variations among NVIDIA GPUs", ICPP 2022**：
  在 856 顆 P100/V100/A100 上跑 48 個 kernel，晶片間的效能與功耗效率差異不可忽略。這支持我們在同一節點上做 ABBA 對照。
  https://doi.org/10.1145/3545008.3545084（Crossref 與 Semantic Scholar 摘要均已核對）
- **Karimi, Maiterth, Shin, Sattar, Lu, Wang, "Exploring the Frontiers of Energy Efficiency using Power Management at System Scale", SC24-W, pp. 1835–1844**：
  以 Frontier 三個月的遙測資料評估 DVFS 與功耗上限：部分資源受限的 job 可省最多 8.5% 能量而不掉效能，推估約 1,438 MWh。
  https://doi.org/10.1109/SCW63240.2024.00230（arXiv:2408.01552 與 Crossref 均已核對）
- **Toledo Costa et al., "Characterizing the Impact of GPU Power Management on an Exascale System", SC25-W, pp. 1524–1533**：
  在 Frontier（MI250X）上比較頻率上限與功耗上限：頻率上限可在不掉效能下提升最多 13.2% 的能效，功耗上限則較適合單節點或突發型負載。
  https://doi.org/10.1145/3731599.3767702（ORNL 出版品頁面與 Crossref 均已核對）
- **Yang, Adamek, Armour, "Accurate and Convenient Energy Measurements for GPUs: A Detailed Study of NVIDIA GPU's Built-In Power Sensor", SC24, pp. 1–17**
  （arXiv 版標題為 "Part-time Power Measurements: nvidia-smi's Lack of Attention"）：在 A100/H100 上，nvidia-smi 只取樣約 25% 的執行時間；
  依他們建議的做法，能耗量測誤差平均可降 35%。**我們的功耗數據來自 nvidia-smi，論文應引用這篇，並說明採用長時間穩態視窗
  與重複的 ABBA 量測。** https://doi.org/10.1109/SC41406.2024.00028（arXiv:2312.02741 與 Crossref 均已核對）
- **Grace Hopper（GH200）相關**：
  - **Hernandez et al., "Preliminary Study on Fine-Grained Power and Energy Measurements on Grace Hopper GH200 with Open-Source Performance Tools", HPC Asia 2025 Workshops, pp. 11–22**：
    整合 hwmon、PAPI、Score-P，以 50 ms 解析度量測 GH200 的 Grace CPU、Hopper GPU 與 I/O 功耗。
    https://doi.org/10.1145/3703001.3724383（ORNL 頁面與 Crossref 均已核對）
  - **Patrou et al., "Power-Capping Metric Evaluation for Improving Energy Efficiency in HPC Applications", ISC 2025 EESP Workshop（LNCS, pp. 231–244）**：
    在 GH200 上以超級晶片功耗上限（預設 1,000 W 即不設限，最低 200 W）配合 automatic power steering 評估 LSMS 各 kernel。
    文中描述：功耗先分給 CPU，剩餘的轉給 GPU，以 DVFS 執行上限；GPU 功耗接近上限時會降 GPU 時脈，計算密集的負載受影響較大。
    計算密集的 GEMM kernel 在約 900 W 時最佳，記憶體密集的 kernel 約 300 W 就最佳。
    **這與我們「模組上限→GPU 時脈」的機制一致，可以直接當背景引用。**
    https://doi.org/10.1007/978-3-032-07612-0_18（arXiv:2505.21758 全文與 Crossref 均已核對）
- **H100/H200 的上限曲線（arXiv 預印本，尚未同儕審查）**：
  - **Ujeniya, Eitzinger, Hager, Wellein, "Architectural Trade-offs in the Energy-Efficient Era: A Comparative Study of power-capping NVIDIA H100 and H200", arXiv:2604.11391（2026）**：
    以 DGEMM 與 TheBandwidthBenchmark 比較不同功耗上限下的表現：計算密集時 H100 略佳，記憶體密集時 H200（HBM3e）較佳。
    https://arxiv.org/abs/2604.11391
  - **Afzal, Li, Panzlaff, "Modeling and Chasing the Energy-Efficiency Sweet Spots in Modern GPUs", arXiv:2607.00819（2026）**：
    以 A40/A100/H100/H200 搭配 GROMACS、AMBER、FIRESTARTER，把 GPU 功耗拆成線性與非線性成分，找出能效轉折頻率；
    結論是對低於 TDP 的負載，降頻比設功耗上限有效。https://arxiv.org/abs/2607.00819
- **Desai, Shah, Alavani, Saha, Sarkar, "Adaptive GPU Power Capping: Balancing Energy Efficiency, Thermal Control and Performance", HPDC 2025 poster**：
  在 RTX 4000 Ada 上以樹模型即時預測最佳上限，省能 12.9%、降溫 11.4%，效能損失 2.7%。
  https://hpdc.sci.utah.edu/2025/assets/slides/posters/desai.pdf（海報 PDF 已核對；ACM DOI 10.1145/3731545.3735119 只在搜尋結果看到，未查證）
- **學生叢集競賽（SCC）的功耗限制**（官方規則頁，非論文）：
  - SC25 SCC：整個叢集（含網路與冷卻）不得超過 10,000 W，任何時刻超過 11,000 W 即自動失格。
    https://sc25.supercomputing.org/students/student-cluster-competition/（已核對）
  - ISC SCC：ISC14 規定 3,000 W（單一電路約 13 A）；目前規則頁（ISC27）為兩條電路合計 6,000 W，超過會扣分或取消提交資格。
    https://www.hpcadvisorycouncil.com/events/2014/isc14-student-cluster-competition/rules.php 、
    https://www.hpcadvisorycouncil.com/events/student-cluster-competition/rules.php（皆已核對）

## 3. 混合精度*儲存*與 memory accessor

**共同特徵與我們的差異**：這條路線利用記憶體頻寬與運算的落差，以*較少的 byte* 存放資料（fp16/fp32/客製或定點格式），
再以高精度運算，收益是頻寬與容量。我們的做法：**byte 數與資料排列都不變**（FP64 仍是 8 bytes），
降低的是每個 byte 的漢明權重與翻轉率，也就是每次搬移或運算的能量。兩者可以疊加，例如在 HPL-MxP 的 BF16 儲存上再清尾數。
Bit Grooming 與 Klöwer 的 bit-information 壓縮會產生與我們相同的「低位為 0」位元型態，但目的是讓無損壓縮更有效，並未探討運算或搬移的功耗。

- **Higham & Mary, "Mixed precision algorithms in numerical linear algebra", Acta Numerica 31:347–414, 2022**：
  混合精度數值線性代數的權威綜述，涵蓋矩陣乘法、分解、線性系統、最小平方、特徵值等，包含迭代修正與 adaptive precision。
  https://doi.org/10.1017/S0962492922000022（Cambridge Core 頁面已核對）
- **Abdelfattah et al.（21 位作者）, "A survey of numerical linear algebra methods utilizing mixed-precision arithmetic", IJHPCA 35(4):344–369, 2021**：
  ECP 的混合精度綜述。https://doi.org/10.1177/10943420211003313（Crossref 與 arXiv:2007.06674 均已核對）
- **Anzt, Dongarra, Flegar, Higham, Quintana-Ortí, "Adaptive precision in block-Jacobi preconditioning for iterative sparse linear system solvers", CCPE 31(6):e4460, 2019**：
  依條件數把每個對角 block 存成 half、single 或 double，運算一律用 double，以減少資料搬移。
  論文的時間與能耗模型假設成本和位元長度成正比。https://doi.org/10.1002/cpe.4460（MIMS EPrints 與 Crossref 均已核對）
- **Anzt, Flegar, Grützmacher, Quintana-Ortí, "Toward a modular precision ecosystem for high-performance computing", IJHPCA 33(6):1069–1078, 2019**：
  提出把儲存精度與運算精度分開的理念，以 block-Jacobi 與 PageRank 示範。https://doi.org/10.1177/1094342019846547（Crossref 已核對）
- **Aliaga, Anzt, Grützmacher, Quintana-Ortí, Tomás, "Compressed basis GMRES on high-performance graphics processing units", IJHPCA 37(2):82–100, 2023**：
  以 Ginkgo 的 memory accessor 用浮點或定點的低位元格式存放 Krylov 基底，在 V100 上比全 double 的 GMRES 最多快 50%。
  https://doi.org/10.1177/10943420221115140（arXiv:2009.12101 已核對）
- **Grützmacher, Anzt, Quintana-Ortí, "Using Ginkgo's memory accessor for improving the accuracy of memory-bound low precision BLAS", SPE 53(1):81–98, 2023**：
  accessor 在存取時即時壓縮與解壓，讓記憶體受限的程式提高 roofline，或讓低精度資料以高精度運算以提升精度。
  https://doi.org/10.1002/spe.3041（ICL 頁面與 Semantic Scholar 摘要均已核對）
- **Graillat, Jézéquel, Mary, Molina, "Adaptive Precision Sparse Matrix–Vector Product and Its Application to Krylov Solvers", SISC 46(1):C30–C56, 2024**：
  依元素大小分桶，小的元素用較低精度，SpMV 最多快 2.5 倍且不損精度。https://doi.org/10.1137/22M1522619（Crossref 已核對）
- **Zender, "Bit Grooming: statistically accurate precision-preserving quantization with compression", GMD 9:3199–3211, 2016**：
  相鄰的值輪流把最低有效位元設為 0 或 1，保留指定精度且統計上無偏，儲存量減少 25–80%。
  https://doi.org/10.5194/gmd-9-3199-2016（GMD 頁面已核對）
- **Klöwer et al., "Compressing atmospheric data into its real information content", Nat. Comput. Sci. 1:713–724, 2021**：
  以位元資訊量判斷哪些位元不帶資訊，把這些位元捨入為 0 後再壓縮。CAMS 資料相對 64-bit 壓縮 17 倍，保留 99% 資訊。
  https://doi.org/10.1038/s43588-021-00156-2（Semantic Scholar 摘要已核對；另見 `references.md`）
- **Haidar, Tomov, Dongarra, Higham, SC18**：FP16 tensor core 的迭代修正最多加速 4 倍，是 HPL-MxP 的演算法基礎。
  這屬於「運算精度」的混合精度，摘要沒有提能耗。https://doi.org/10.1109/SC.2018.00050（Semantic Scholar 摘要已核對）

## 4. 資料搬移的能耗取決於資料值（位元翻轉、DBI、編碼）

**共同特徵與我們的差異**：電路與架構研究早已確立兩點：匯流排的動態功耗與位元翻轉數成正比；在 POD 介面下，
傳 0 會有 DC 終端電流。因此有了 bus-invert、DBI-DC/AC 與 BD-encoding 等編碼，以及 toggle-aware compression。
這些研究大多是*硬體或編碼提案*，以模擬評估，或在 DDR 模組上實測。我們的貢獻：在現成的 GH200（HBM3）上，
只改軟體端的資料值，端到端量到記憶體複製與 GEMM 的功耗與能耗下降，並連結到功耗上限下的時脈效應。
目前沒有找到在 GPU 上實測「記憶體受限 kernel 的功耗隨資料值變化」的同儕審查論文，這是我們可以宣稱的空白。

- **Stan & Burleson, "Bus-Invert Coding for Low-Power I/O", IEEE TVLSI 3(1):49–58, 1995**：CMOS 的動態功耗與節點活動因子成正比；
  bus-invert 編碼可讓 I/O 峰值功耗降 50%、平均功耗最多降 25%。http://www.cs.columbia.edu/~cs4823/handouts/stan-burleson-tvlsi-95.pdf（PDF 全文已核對）
- **Hollis, "Data Bus Inversion in High-Speed Memory Applications", IEEE TCAS-II 56(4):300–304, 2009**：高速記憶體介面的 DBI 演算法比較。
  https://doi.org/10.1109/TCSII.2009.2015395（Crossref 已核對；摘要內容未能開啟，細節未查證）
- **Lucas, Lal, Juurlink, "Optimal DC/AC Data Bus Inversion Coding", DATE 2018**：GDDR5 與 DDR4 採 POD 介面，
  傳 0 時終端電阻有 DC 電流、傳 1 沒有；DBI-DC 讓 0 的數量最少，DBI-AC 讓翻轉最少。兩者同時考慮可再省最多 6% 的介面功耗。
  https://doi.org/10.23919/DATE.2018.8342169（作者 PDF 與 Crossref 均已核對）
- **Seol et al., "Energy Efficient Data Encoding in DRAM Channels Exploiting Data Value Similarity", ISCA 2016, pp. 719–730**：
  BD-Encoding 利用相鄰資料字的相似性降低漢明權重，同時減少終端能耗與切換能耗（DDR4、GDDR5 的 POD 與 LPDDR4 的 LVSTL）。
  作者投影片提到終端能耗減少 58.3%、切換能耗減少 45.3%。https://doi.org/10.1109/ISCA.2016.68（Crossref 與作者 ISCA 投影片均已核對）
- **Chatterjee, O'Connor, Lee, Johnson, Keckler, Rhu, Dally, "Architecting an Energy-Efficient DRAM System for GPUs", HPCA 2017, pp. 73–84**：
  IDD4R/W 規格假設 50% 翻轉率；以他們的 HBM 模型計算，翻轉率 0% 時 column energy 約 1.5 pJ/bit，100% 時約 5.7 pJ/bit。
  資料路徑中與翻轉無關的部分約 1.48 pJ/b，其餘在 50% 翻轉率時約 2.85 pJ/b。論文採用「HBM 所用的 DBI-AC」，
  並以靜態資料重排降低翻轉率。https://doi.org/10.1109/HPCA.2017.58（NVIDIA Research PDF 全文與 Crossref 均已核對）
- **O'Connor et al., "Fine-Grained DRAM: Energy-Efficient DRAM for Extreme Bandwidth Systems", MICRO 2017, pp. 41–54**：
  HBM2 每位元存取約 3.92 pJ，其中晶片內資料路徑約 2.24 pJ/b，interposer I/O 約 0.3 pJ/b（皆依實際應用的翻轉率計）。
  post-GSA 資料路徑與 I/O 能耗取決於翻轉率，pre-GSA 則不受影響，因此不同應用的資料搬移能耗會隨資料翻轉率而不同。
  https://doi.org/10.1145/3123939.3124545（作者 PDF 全文與 Crossref 均已核對）
- **Ghose et al., "What Your DRAM Power Models Are Not Telling You: Lessons from a Detailed Experimental Study", POMACS 2(3), 2018（SIGMETRICS 2018）**：
  實測 50 條 DDR3L 模組，發現 DRAM 功耗強烈取決於讀寫的資料值：讀取電流隨 cache line 中 1 的個數增加而上升，
  寫入電流則下降（vendor A 最多差 434 mA 與 311 mA）；電流也隨翻轉位元數線性增加。據此建立 VAMPIRE 模型。
  https://doi.org/10.1145/3224419（arXiv:1807.05102 全文與 Crossref 均已核對）
- **Pekhimenko, Bolotin, Vijaykumar, Mutlu, Mowry, Keckler, "A Case for Toggle-Aware Compression for GPU Systems", HPCA 2016, pp. 188–200**：
  壓縮減少了 byte 數，卻讓每個位元的熵變高，GPU 片上與片外匯流排的翻轉數增加 20% 到 2.2 倍，動態能耗隨之上升。
  論文提出 Energy Control 與 Metadata Consolidation。**這是「減少 byte 數」與「減少翻轉」並非同一件事的關鍵證據，
  可以同時支撐第 3、4 節的論點。** http://www.cs.utexas.edu/~skeckler/pubs/HPCA_2016_Compression.pdf（PDF 全文已核對；
  DOI 10.1109/HPCA.2016.7446064 取自書目，未另外核對）
- **Rhu, O'Connor, Chatterjee, Pool, Kwon, Keckler, "Compressing DMA Engine: Leveraging Activation Sparsity for Training Deep Neural Networks", HPCA 2018, pp. 78–91**：
  利用 activation 中的零值（zero-value compression 等），把 CPU↔GPU 傳輸平均壓縮 2.6 倍，減少的是 byte 數。
  https://doi.org/10.1109/HPCA.2018.00017（KAIST 出版品頁面已核對）
- **Peltekis, Filippas, Dimitrakopoulos, Nicopoulos, "Low-Power Data Streaming in Systolic Arrays with Bus-Invert Coding and Zero-Value Clock Gating", MOCAST 2023**：
  在 BF16 systolic array 內，對資料流做 bus-invert 編碼並對零值做 clock gating，整體動態功耗降 6.2–9.4%。
  這證明運算陣列內部的資料流功耗也取決於位元型態。https://doi.org/10.1109/MOCAST57943.2023.10176467（arXiv:2304.12691 已核對）
- **HBM3 沿用 DBI(ac)**：Synopsys 的技術文章（Brett Murdock, 2022）寫道「Some things are carried forward from HBM2E into HBM3,
  such as DBI(ac)」。屬於業界文章，非論文。https://www.synopsys.com/articles/hbm3-ip-dwtb.html（已核對）
- **計算端的資料相依功耗**（GEMM，見 `references.md`）：Gregersen et al. SC24-W（https://doi.org/10.1109/SCW63240.2024.00235，
  Crossref 已核對）與 Bhalachandra et al. arXiv:2212.08805（https://arxiv.org/abs/2212.08805，已核對）。
  兩者都只測 GEMM，沒有測記憶體受限的 kernel。

---

## 可引用文獻（皆已查證）

### 精度調校
- M. O. Lam, J. K. Hollingsworth, B. R. de Supinski, M. P. LeGendre. *Automatically adapting programs for mixed-precision floating-point computation.* ICS 2013, pp. 369–378. doi:10.1145/2464996.2465018
- C. Rubio-González, C. Nguyen, H. D. Nguyen, J. Demmel, W. Kahan, K. Sen, D. H. Bailey, C. Iancu, D. Hough. *Precimonious: Tuning Assistant for Floating-Point Precision.* SC13. doi:10.1145/2503210.2503296
- C. Rubio-González, C. Nguyen, B. Mehne, K. Sen, J. Demmel, W. Kahan, C. Iancu, W. Lavrijsen, D. H. Bailey, D. Hough. *Floating-Point Precision Tuning Using Blame Analysis.* ICSE 2016. doi:10.1145/2884781.2884850
- W.-F. Chiang, M. Baranowski, I. Briggs, A. Solovyev, G. Gopalakrishnan, Z. Rakamarić. *Rigorous Floating-Point Mixed-Precision Tuning.* POPL 2017, pp. 300–315. doi:10.1145/3009837.3009846
- H. Guo, C. Rubio-González. *Exploiting Community Structure for Floating-Point Precision Tuning.* ISSTA 2018. doi:10.1145/3213846.3213862
- H. Menon, M. O. Lam, D. Osei-Kuffuor, M. Schordan, S. Lloyd, K. Mohror, J. Hittinger. *ADAPT: Algorithmic Differentiation Applied to Floating-Point Precision Tuning.* SC18, pp. 614–626. doi:10.1109/SC.2018.00051
- I. Laguna, P. C. Wood, R. Singh, S. Bagchi. *GPUMixer: Performance-Driven Floating-Point Tuning for GPU Scientific Applications.* ISC High Performance 2019 (LNCS). doi:10.1007/978-3-030-20656-7_12
- P. V. Kotipalli, R. Singh, P. Wood, I. Laguna, S. Bagchi. *AMPT-GA: Automatic Mixed Precision Floating Point Tuning for GPU Applications.* ICS 2019, pp. 160–170. doi:10.1145/3330345.3330360
- S. Kang, K. Choi, Y. Park. *PreScaler: An Efficient System-Aware Precision Scaling Framework on Heterogeneous Systems.* CGO 2020, pp. 280–292. doi:10.1145/3368826.3377917
- H. Brunie, C. Iancu, K. Z. Ibrahim, P. Brisk, B. Cook. *Tuning Floating-Point Precision Using Dynamic Program Information and Temporal Locality.* SC20. doi:10.1109/SC41405.2020.00054
- Y. Wang, C. Rubio-González. *Predicting Performance and Accuracy of Mixed-Precision Programs for Precision Tuning.* ICSE 2024. doi:10.1145/3597503.3623338
- F. Hoerold, I. R. Ivanov, A. Dhruv, W. S. Moses, A. Dubey, M. Wahib, J. Domke. *RAPTOR: Practical Numerical Profiling of Scientific Applications.* SC25, pp. 661–680. doi:10.1145/3712285.3759810
- Z. Li, Y. Wu, Y. Zhang. *Multi-Objective Optimization for Floating Point Mix-Precision Tuning.* ISLPED 2023. doi:10.1109/ISLPED58423.2023.10244539
- G. Tagliavini, S. Mach, D. Rossi, A. Marongiu, L. Benini. *A Transprecision Floating-Point Platform for Ultra-Low Power Computing.* DATE 2018, pp. 1051–1056. doi:10.23919/DATE.2018.8342167
- J. Y. F. Tong, D. Nagle, R. A. Rutenbar. *Reducing Power by Optimizing the Necessary Precision/Range of Floating-Point Arithmetic.* IEEE TVLSI 8(3):273–286, 2000. doi:10.1109/92.845894
- A. Dawson, P. D. Düben. *rpe v5: an emulator for reduced floating-point precision in large numerical simulations.* Geosci. Model Dev. 10:2221–2230, 2017. doi:10.5194/gmd-10-2221-2017
- N. J. Higham, S. Pranesh. *Simulating Low Precision Floating-Point Arithmetic.* SIAM J. Sci. Comput. 41(5):C585–C602, 2019. doi:10.1137/19M1251308
- A. Adjé, D. Ben Khalifa, M. Martel. *Fast and Efficient Bit-Level Precision Tuning.* arXiv:2103.05241, 2021. https://arxiv.org/abs/2103.05241
- X. Chen, T. Hilaire, F. Jézéquel. *Floating-point autotuning with customized precisions.* arXiv:2606.08339, 2026. https://arxiv.org/abs/2606.08339
- E. Carson, X. Chen. *Precision autotuning for linear solvers via contextual bandit-based RL.* arXiv:2601.00728, 2026. https://arxiv.org/abs/2601.00728
- P. V. Argade, D. N. Peroni (Advanced Micro Devices). *Variable Precision Computing System.* US Patent 11,455,766 B2, granted 2022-09-27. https://patents.google.com/patent/US11455766B2/en

### 功耗上限
- H. David, E. Gorbatov, U. R. Hanebutte, R. Khanna, C. Le. *RAPL: Memory Power Estimation and Capping.* ISLPED 2010, pp. 189–194. doi:10.1145/1840845.1840883
- B. Rountree, D. H. Ahn, B. R. de Supinski, D. K. Lowenthal, M. Schulz. *Beyond DVFS: A First Look at Performance under a Hardware-Enforced Power Bound.* IPDPSW 2012, pp. 947–953. doi:10.1109/IPDPSW.2012.116
- T. Patki, D. K. Lowenthal, B. Rountree, M. Schulz, B. R. de Supinski. *Exploring Hardware Overprovisioning in Power-Constrained, High Performance Computing.* ICS 2013, pp. 173–182. doi:10.1145/2464996.2465009
- T. Patki, D. K. Lowenthal, A. Sasidharan, M. Maiterth, B. L. Rountree, M. Schulz, B. R. de Supinski. *Practical Resource Management in Power-Constrained, High Performance Computing.* HPDC 2015, pp. 121–132. doi:10.1145/2749246.2749262
- Y. Inadomi, T. Patki, K. Inoue, M. Aoyagi, B. Rountree, M. Schulz, D. Lowenthal, Y. Wada, K. Fukazawa, M. Ueda, M. Kondo, I. Miyoshi. *Analyzing and Mitigating the Impact of Manufacturing Variability in Power-Constrained Supercomputing.* SC15. doi:10.1145/2807591.2807638
- A. Haidar, H. Jagode, P. Vaccaro, A. YarKhan, S. Tomov, J. Dongarra. *Investigating Power Capping toward Energy-Efficient Scientific Applications.* Concurrency Computat. Pract. Exper. 31(6):e4485, 2019. doi:10.1002/cpe.4485
- A. Krzywaniak, P. Czarnul. *Performance/Energy Aware Optimization of Parallel Applications on GPUs Under Power Capping.* PPAM 2019, LNCS 12044, pp. 123–133, 2020. doi:10.1007/978-3-030-43222-5_11
- J. McDonald, B. Li, N. Frey, D. Tiwari, V. Gadepally, S. Samsi. *Great Power, Great Responsibility: Recommendations for Reducing Energy for Training Language Models.* Findings of NAACL 2022, pp. 1962–1970. doi:10.18653/v1/2022.findings-naacl.151
- D. Zhao, S. Samsi, J. McDonald, B. Li, D. Bestor, M. Jones, D. Tiwari, V. Gadepally. *Sustainable Supercomputing for AI: GPU Power Capping at HPC Scale.* SoCC 2023, pp. 588–596. doi:10.1145/3620678.3624793
- P. Patel, E. Choukse, C. Zhang, Í. Goiri, B. Warrier, N. Mahalingam, R. Bianchini. *Characterizing Power Management Opportunities for LLMs in the Cloud.* ASPLOS 2024, pp. 207–222. doi:10.1145/3620666.3651329
- K. Yoshida, R. Sageyama, S. Miwa, H. Yamaki, H. Honda. *Analyzing Performance and Power-Efficiency Variations among NVIDIA GPUs.* ICPP 2022. doi:10.1145/3545008.3545084
- A. M. Karimi, M. Maiterth, W. Shin, N. S. Sattar, H. Lu, F. Wang. *Exploring the Frontiers of Energy Efficiency using Power Management at System Scale.* SC24-W, pp. 1835–1844. doi:10.1109/SCW63240.2024.00230
- M. Toledo Costa, A. Georgiadou, J. B. White III, B. Villasenor Alvarez, J. Polo, W. Shin, P. O. A. Navaux, B. Messer, A. F. Lorenzon. *Characterizing the Impact of GPU Power Management on an Exascale System.* SC25-W, pp. 1524–1533. doi:10.1145/3731599.3767702
- Z. Yang, K. Adamek, W. Armour. *Accurate and Convenient Energy Measurements for GPUs: A Detailed Study of NVIDIA GPU's Built-In Power Sensor.* SC24, pp. 1–17. doi:10.1109/SC41406.2024.00028
- O. Hernandez, T. Wang, W. Elwasif, F. Spiga, F. Tartaglione, M. Eisenbach, R. Miller. *Preliminary Study on Fine-Grained Power and Energy Measurements on Grace Hopper GH200 with Open-Source Performance Tools.* HPC Asia 2025 Workshops, pp. 11–22. doi:10.1145/3703001.3724383
- M. Patrou, T. Wang, W. Elwasif, M. Eisenbach, R. Miller, W. Godoy, O. Hernandez. *Power-Capping Metric Evaluation for Improving Energy Efficiency in HPC Applications.* ISC High Performance 2025 Workshops (EESP), LNCS, pp. 231–244. doi:10.1007/978-3-032-07612-0_18
- A. Ujeniya, J. Eitzinger, G. Hager, G. Wellein. *Architectural Trade-offs in the Energy-Efficient Era: A Comparative Study of power-capping NVIDIA H100 and H200.* arXiv:2604.11391, 2026. https://arxiv.org/abs/2604.11391
- A. Afzal, M. M. Li, M. Panzlaff. *Modeling and Chasing the Energy-Efficiency Sweet Spots in Modern GPUs.* arXiv:2607.00819, 2026. https://arxiv.org/abs/2607.00819
- T. Desai, J. Shah, G. Alavani, S. Saha, S. Sarkar. *Adaptive GPU Power Capping: Balancing Energy Efficiency, Thermal Control and Performance.* HPDC 2025 poster. https://hpdc.sci.utah.edu/2025/assets/slides/posters/desai.pdf
- SC25 Student Cluster Competition rules（10,000 W）. https://sc25.supercomputing.org/students/student-cluster-competition/
- ISC Student Cluster Competition rules（ISC14 3,000 W；現行 6,000 W）. https://www.hpcadvisorycouncil.com/events/2014/isc14-student-cluster-competition/rules.php ; https://www.hpcadvisorycouncil.com/events/student-cluster-competition/rules.php

### 混合精度儲存
- N. J. Higham, T. Mary. *Mixed precision algorithms in numerical linear algebra.* Acta Numerica 31:347–414, 2022. doi:10.1017/S0962492922000022
- A. Abdelfattah, H. Anzt, E. G. Boman, E. Carson, T. Cojean, et al. *A survey of numerical linear algebra methods utilizing mixed-precision arithmetic.* IJHPCA 35(4):344–369, 2021. doi:10.1177/10943420211003313
- H. Anzt, J. Dongarra, G. Flegar, N. J. Higham, E. S. Quintana-Ortí. *Adaptive precision in block-Jacobi preconditioning for iterative sparse linear system solvers.* Concurrency Computat. Pract. Exper. 31(6):e4460, 2019. doi:10.1002/cpe.4460
- H. Anzt, G. Flegar, T. Grützmacher, E. S. Quintana-Ortí. *Toward a modular precision ecosystem for high-performance computing.* IJHPCA 33(6):1069–1078, 2019. doi:10.1177/1094342019846547
- J. I. Aliaga, H. Anzt, T. Grützmacher, E. S. Quintana-Ortí, A. E. Tomás. *Compressed basis GMRES on high-performance graphics processing units.* IJHPCA 37(2):82–100, 2023. doi:10.1177/10943420221115140
- T. Grützmacher, H. Anzt, E. S. Quintana-Ortí. *Using Ginkgo's memory accessor for improving the accuracy of memory-bound low precision BLAS.* Softw. Pract. Exper. 53(1):81–98, 2023. doi:10.1002/spe.3041
- S. Graillat, F. Jézéquel, T. Mary, R. Molina. *Adaptive Precision Sparse Matrix–Vector Product and Its Application to Krylov Solvers.* SIAM J. Sci. Comput. 46(1):C30–C56, 2024. doi:10.1137/22M1522619
- C. S. Zender. *Bit Grooming: statistically accurate precision-preserving quantization with compression, evaluated in the netCDF Operators (NCO, v4.4.8+).* Geosci. Model Dev. 9:3199–3211, 2016. doi:10.5194/gmd-9-3199-2016
- M. Klöwer, M. Razinger, J. J. Dominguez, P. D. Düben, T. N. Palmer. *Compressing atmospheric data into its real information content.* Nat. Comput. Sci. 1:713–724, 2021. doi:10.1038/s43588-021-00156-2
- A. Haidar, S. Tomov, J. Dongarra, N. J. Higham. *Harnessing GPU Tensor Cores for Fast FP16 Arithmetic to Speed up Mixed-Precision Iterative Refinement Solvers.* SC18. doi:10.1109/SC.2018.00050

### 資料相依的資料搬移能耗
- M. R. Stan, W. P. Burleson. *Bus-Invert Coding for Low-Power I/O.* IEEE TVLSI 3(1):49–58, 1995. http://www.cs.columbia.edu/~cs4823/handouts/stan-burleson-tvlsi-95.pdf
- T. M. Hollis. *Data Bus Inversion in High-Speed Memory Applications.* IEEE TCAS-II 56(4):300–304, 2009. doi:10.1109/TCSII.2009.2015395
- J. Lucas, S. Lal, B. Juurlink. *Optimal DC/AC Data Bus Inversion Coding.* DATE 2018. doi:10.23919/DATE.2018.8342169
- H. Seol, W. Shin, J. Jang, J. Choi, J. Suh, L.-S. Kim. *Energy Efficient Data Encoding in DRAM Channels Exploiting Data Value Similarity.* ISCA 2016, pp. 719–730. doi:10.1109/ISCA.2016.68
- N. Chatterjee, M. O'Connor, D. Lee, D. R. Johnson, S. W. Keckler, M. Rhu, W. J. Dally. *Architecting an Energy-Efficient DRAM System for GPUs.* HPCA 2017, pp. 73–84. doi:10.1109/HPCA.2017.58
- M. O'Connor, N. Chatterjee, D. Lee, J. Wilson, A. Agrawal, S. W. Keckler, W. J. Dally. *Fine-Grained DRAM: Energy-Efficient DRAM for Extreme Bandwidth Systems.* MICRO 2017, pp. 41–54. doi:10.1145/3123939.3124545
- S. Ghose, A. G. Yağlıkçı, R. Gupta, D. Lee, K. Kudrolli, W. X. Liu, H. Hassan, K. K. Chang, N. Chatterjee, A. Agrawal, M. O'Connor, O. Mutlu. *What Your DRAM Power Models Are Not Telling You: Lessons from a Detailed Experimental Study.* Proc. ACM Meas. Anal. Comput. Syst. (POMACS) 2(3), 2018 (SIGMETRICS 2018). doi:10.1145/3224419
- G. Pekhimenko, E. Bolotin, N. Vijaykumar, O. Mutlu, T. C. Mowry, S. W. Keckler. *A Case for Toggle-Aware Compression for GPU Systems.* HPCA 2016, pp. 188–200. http://www.cs.utexas.edu/~skeckler/pubs/HPCA_2016_Compression.pdf
- M. Rhu, M. O'Connor, N. Chatterjee, J. Pool, Y. Kwon, S. W. Keckler. *Compressing DMA Engine: Leveraging Activation Sparsity for Training Deep Neural Networks.* HPCA 2018, pp. 78–91. doi:10.1109/HPCA.2018.00017
- C. Peltekis, D. Filippas, G. Dimitrakopoulos, C. Nicopoulos. *Low-Power Data Streaming in Systolic Arrays with Bus-Invert Coding and Zero-Value Clock Gating.* MOCAST 2023. doi:10.1109/MOCAST57943.2023.10176467
- T. Gregersen, P. Patel, E. Choukse. *Input-Dependent Power Usage in GPUs.* SC24-W, pp. 1872–1877. doi:10.1109/SCW63240.2024.00235
- S. Bhalachandra, B. Austin, S. Williams, N. J. Wright. *Understanding the Impact of Input Entropy on FPU, CPU, and GPU Power.* arXiv:2212.08805, 2022. https://arxiv.org/abs/2212.08805
- B. Murdock (Synopsys). *What Designers Need to Know About HBM3.* 2022（業界文章）. https://www.synopsys.com/articles/hbm3-ip-dwtb.html

---

## 未查證／待補

- **JEDEC JESD235（HBM 標準）中 DBI-AC 的規定**：沒有開啟標準本文（付費），目前只能經由 Chatterjee et al. HPCA 2017 的引用
  與 Synopsys 的文章間接得知。**GH200 的 HBM3 在實際運作時是否啟用 DBI，也未查證。**
- **CMU-SAFARI HBM-Power（MICRO'26 artifact）**：GitHub 上有 36 個 HBM2 stack 的功耗特性量測，另含 H200（HBM3E）的量測
  （https://github.com/CMU-SAFARI/HBM-Power 已開啟），但對應論文的標題、作者，以及是否分析資料相依性都未查證。
  這可能是最接近「GPU HBM 資料相依功耗」的實測，論文正式出版後應補查。
- Desai et al. HPDC 2025 的 ACM DOI（10.1145/3731545.3735119）只在搜尋結果中看到。
- Adjé/Ben Khalifa/Martel 的 SAS 2021 出版資訊（DOI 10.1007/978-3-030-88806-0_1）只在搜尋結果中看到。
- Hollis TCAS-II 2009 的摘要與內容（只核對了 Crossref 書目）。
- Pekhimenko HPCA 2016 的 DOI（全文 PDF 已核對，DOI 未另外查 Crossref）。
- Krzywaniak & Czarnul PPAM 2019 使用的 GPU 型號。
- 有搜尋但沒有找到：**在 GPU 上實測記憶體受限 kernel（STREAM 或 memcpy）功耗隨資料值變化的同儕審查論文**，
  以及其他「軟體清尾數→功耗上限下時脈上升」的 HPC 論文。這兩點目前看來是我們可以宣稱的新貢獻，
  但投稿前應再用 IEEE Xplore／ACM DL 做一次關鍵字檢索（data-dependent power、bit toggling、HBM power data pattern、
  mantissa truncation power）。

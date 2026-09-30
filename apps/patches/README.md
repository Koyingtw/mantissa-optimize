# 第三方軟體的尾數截斷修改

這些軟體直接使用上游原始碼（不是 NTHU-SC 的 fork），修改不在上游 repo 內提交，而是存成 patch。
在對應的基準提交上 `git -C apps/<name> apply ../patches/<name>.patch` 即可重現。建置方式見 `apps/建置日誌.md`。

| 軟體 | 基準提交 | 修改內容 |
|---|---|---|
| q-e | 9f93ddec427d2b9a45bb72d828c6d324f62fcabd（qe-7.6，submodule） | Davidson（cegterg／regterg）程式內截斷與依收斂門檻自動調整位元（`QE_MANT_ADAPT`、`QE_MANT_ROUND`、`QE_MANT_MIN`） |
| gpu4pyscf | 5b284c258a4260baef80e3d150b4e7a81a9dbd57（v1.8.1，submodule） | 建置設定（CMake） |
| lammps | 9c5ab448c78a14fd534619622162ba418d6a1fb1（stable_22Jul2025_update6） | KOKKOS EAM／LJ 的係數截斷（`MANT_BITS`、`MANT_ROUND`） |
| mfem | c362dea2470d2d27be15df6e55ae1f3c85d031f3 | 免組裝擴散／質量積分器的求積點資料截斷 |
| sw4 | 52b51c9253c0c30b608744ef17ee7a28661939bc | `setupRun.C` 在設定階段截斷材料參數 |

未設定 `MANT_BITS`（或 ≥ 52）時，四個程式的結果都與原版逐位相同或在重跑雜訊內。

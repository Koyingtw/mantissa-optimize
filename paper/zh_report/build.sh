#!/bin/bash
# 以 tectonic（XeLaTeX + BibTeX）編譯 report.tex，輸出 report.pdf；沒有 tectonic 時改用系統的 xelatex＋bibtex。
set -euo pipefail
cd "$(dirname "$0")"
[[ -f fonts/NotoSerifCJKtc-Regular.otf ]] || ./fetch_fonts.sh
TECTONIC=${TECTONIC:-$(command -v tectonic || echo ~/.local/bin/tectonic)}
if [[ -x $TECTONIC ]]; then
  "$TECTONIC" -X compile --keep-logs report.tex
else  # MI350X 節點：apt 的 TeX Live 2023（需 texlive-lang-chinese）
  xelatex -interaction=nonstopmode -halt-on-error report.tex > /dev/null || { grep -A3 "^! " report.log | head -20; exit 1; }
  bibtex report > /dev/null || true
  xelatex -interaction=nonstopmode report.tex > /dev/null
  xelatex -interaction=nonstopmode report.tex > /dev/null
  rm -f report.{aux,bbl,blg,out,toc,lof,lot}
fi
grep -E "Overfull|Underfull .*badness 10000|undefined|Missing character" report.log || echo "沒有過寬的行、未定義的參照或缺字"

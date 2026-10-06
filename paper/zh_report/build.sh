#!/bin/bash
# 以 tectonic（XeLaTeX + BibTeX）編譯 report.tex，輸出 report.pdf。
set -euo pipefail
cd "$(dirname "$0")"
[[ -f fonts/NotoSerifCJKtc-Regular.otf ]] || ./fetch_fonts.sh
TECTONIC=${TECTONIC:-$(command -v tectonic || echo ~/.local/bin/tectonic)}
"$TECTONIC" -X compile --keep-logs report.tex
grep -E "Overfull|Underfull .*badness 10000|undefined|Missing character" report.log || echo "沒有過寬的行、未定義的參照或缺字"

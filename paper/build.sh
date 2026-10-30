#!/bin/bash
# 以 XeLaTeX + BibTeX 編譯英文稿（MI350X 節點以 apt 安裝 TeX Live 2023；GH200 叢集上用 tectonic）。
#   paper/build.sh [main_v2]      預設 main；輸出 paper/<name>.pdf，並列出未定義參照、缺字與過寬的行
set -euo pipefail
cd "$(dirname "$0")"
NAME=${1:-main}
xelatex -interaction=nonstopmode -halt-on-error "$NAME.tex" > /dev/null || { grep -A3 "^! " "$NAME.log" | head -20; exit 1; }
bibtex "$NAME" > /dev/null || true
xelatex -interaction=nonstopmode "$NAME.tex" > /dev/null
xelatex -interaction=nonstopmode "$NAME.tex" > /dev/null
grep -E "undefined|Missing character|Overfull" "$NAME.log" | grep -v "Font shape .TU/ptm" || echo "沒有未定義的參照、缺字或過寬的行"
grep -E "^Output written" "$NAME.log"
rm -f "$NAME".{aux,bbl,blg,out}

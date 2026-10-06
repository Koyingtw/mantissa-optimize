#!/bin/bash
# 下載本報告使用的字型（皆為 SIL Open Font License）到 fonts/，並以 SHA-256 檢查：
#   Noto Serif／Sans CJK TC（內文與標題）、NotoSansTC 可變字型（gh200/analysis/report_figs_zh.py 畫圖用）。
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p fonts
NOTO=https://github.com/notofonts/noto-cjk/raw/main
while read -r sum name url; do
  f=fonts/$name
  if [[ ! -f $f ]] || ! echo "$sum  $f" | sha256sum -c --quiet - 2>/dev/null; then
    echo "下載 $name"
    curl -fsSL -o "$f" "$url"
  fi
  echo "$sum  $f" | sha256sum -c --quiet - || { echo "$name 的雜湊值不符（上游可能已更新字型）" >&2; exit 1; }
done <<LIST
234301038e76e7c35c43113785024700c4e4fe7bdce1d1fbbc42fca7e6683798 NotoSerifCJKtc-Regular.otf $NOTO/Serif/OTF/TraditionalChinese/NotoSerifCJKtc-Regular.otf
a4441a76dbf56719600c5dcbd5b5e5a068a20944cc41c959487a657133576ee6 NotoSerifCJKtc-Bold.otf $NOTO/Serif/OTF/TraditionalChinese/NotoSerifCJKtc-Bold.otf
dce08bd4fd91aa8aa76ed8fea4b694c2dfb8550f67871e326843212ddbeb88b4 NotoSansCJKtc-Regular.otf $NOTO/Sans/OTF/TraditionalChinese/NotoSansCJKtc-Regular.otf
3ee160e5015106e3ec1a394301df54fa9bbbf8a251519984aec5c0abc50840c0 NotoSansCJKtc-Bold.otf $NOTO/Sans/OTF/TraditionalChinese/NotoSansCJKtc-Bold.otf
864727d210d54f2537bbe23b3a839436c3992af72de9322af5270897246bd44f NotoSansTC.ttf https://github.com/google/fonts/raw/main/ofl/notosanstc/NotoSansTC%5Bwght%5D.ttf
LIST
echo "字型就緒：$(ls fonts | tr '\n' ' ')"

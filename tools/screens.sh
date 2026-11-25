#!/bin/sh
# Screenshot the locally served page at phone, tablet and desktop sizes into .shots/.
# Needs `python3 tools/serve.py` running. Reduced motion so intro animations show their end state.
# Usage: tools/screens.sh [URL]
set -eu
URL=${1:-http://127.0.0.1:8000/}
CHROME=$(command -v google-chrome || command -v chromium || command -v chromium-browser)
OUT=$(dirname "$0")/../.shots
mkdir -p "$OUT"
for size in phone-360x740 phone-390x844 tablet-820x1180 tablet-1180x820 desktop-1440x900; do
  wh=${size#*-}; w=${wh%x*}; h=${wh#*x}
  for mode in fold full; do
    [ "$mode" = full ] && h2=16000 || h2=$h
    timeout 90 "$CHROME" --headless=new --disable-gpu --hide-scrollbars --force-prefers-reduced-motion \
      --virtual-time-budget=6000 --window-size="$w,$h2" --screenshot="$OUT/$size-$mode.png" "$URL" >/dev/null 2>&1
  done
  echo "$OUT/$size-{fold,full}.png"
done

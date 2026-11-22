#!/bin/sh
# Run from any directory with a Python environment containing PyInstaller.
set -eu
cd "$(dirname "$0")/.."
out="${PINKPROCESS_MAC_BUILD_DIR:-$PWD/build/mac-kit}"
app="$out/pinkProcess.app/Contents"
mkdir -p "$app/MacOS" "$app/Resources"
grep -E '^(export )?PYNQ_AUTH0_(DOMAIN|CLIENT_ID)=' .env > "$out/pynq.env" 2>/dev/null || : > "$out/pynq.env"
"${PYTHON:-python3}" -m PyInstaller --noconfirm --clean --name kit-backend --paths "$PWD" \
    --distpath "$out/dist" --workpath "$out/work" --specpath "$out" \
    --add-data "$PWD/pinkprocess/kit.html:pinkprocess" --add-data "$PWD/pinkprocess/static:pinkprocess/static" --add-data "$PWD/datasets:datasets" --add-data "$PWD/paper/main.pdf:paper" --add-data "$out/pynq.env:." mac/kit_backend.py
ditto "$out/dist/kit-backend" "$app/Resources/backend"
cp mac/Info.plist "$app/Info.plist"
cp mac/KitWindow.swift "$out/main.swift"
xcrun swiftc -target arm64-apple-macos13.0 -O "$out/main.swift" mac/SpeechInput.swift mac/WorkbenchOrigin.swift -o "$app/MacOS/pinkProcess"
codesign --force --deep --sign - "$out/pinkProcess.app"
codesign --verify --deep --strict "$out/pinkProcess.app"

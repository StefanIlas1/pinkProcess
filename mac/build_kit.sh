#!/bin/sh
# Run from any directory with a Python environment containing PyInstaller.
set -eu
cd "$(dirname "$0")/.."
out="$PWD/build/mac-kit"
app="$out/pinkProcess.app/Contents"
mkdir -p "$app/MacOS" "$app/Resources"
"${PYTHON:-python3}" -m PyInstaller --noconfirm --clean --name kit-backend --paths "$PWD" \
    --distpath "$out/dist" --workpath "$out/work" --specpath "$out" \
    --add-data "$PWD/pinkprocess/kit.html:pinkprocess" mac/kit_backend.py
ditto "$out/dist/kit-backend" "$app/Resources/backend"
cp mac/Info.plist "$app/Info.plist"
cp mac/KitWindow.swift "$out/main.swift"
xcrun swiftc -O "$out/main.swift" mac/LiveCSV.swift -o "$app/MacOS/pinkProcess"
codesign --force --deep --sign - "$out/pinkProcess.app"
codesign --verify --deep --strict "$out/pinkProcess.app"

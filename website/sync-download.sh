#!/bin/sh
set -e
cd "$(dirname "$0")"
mkdir -p downloads
cp ../build/mac-kit/pinkProcess-Mac-arm64.zip downloads/

#!/bin/sh
set -e
cd "$(dirname "$0")"
mkdir -p downloads
cp ../build/mac-kit/pinkProcess-Mac-arm64.zip downloads/
mkdir -p paper
cp ../paper/main.pdf paper/pynq-process-paper.pdf

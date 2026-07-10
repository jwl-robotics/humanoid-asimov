#!/usr/bin/env bash
# Fetch external assets (not committed): the open-source Asimov MuJoCo model (sparse, ~60 MB
# vs the full ~650 MB repo) and the reference walking trajectory used by walk.py / run_walk.py.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# 1. MuJoCo model (sparse checkout of just sim-model/)
mkdir -p "$ROOT/external"
if [ -d "$ROOT/external/asimov-1/.git" ]; then
  echo "model: already present"
else
  git clone --depth 1 --filter=blob:none --sparse https://github.com/asimovinc/asimov-1.git "$ROOT/external/asimov-1"
  git -C "$ROOT/external/asimov-1" sparse-checkout set sim-model
  echo "model: external/asimov-1/sim-model/xmls/asimov.xml"
fi

# 2. Walking reference trajectory (single file from asimov-mjlab)
mkdir -p "$ROOT/data"
if [ -f "$ROOT/data/walking_1.25Hz_50Hz.csv" ]; then
  echo "walking csv: already present"
else
  URL="https://raw.githubusercontent.com/asimovinc/asimov-mjlab/main/new_imitation_data_walking_1.25Hz_50Hz.csv"
  if curl -fsSL "$URL" -o "$ROOT/data/walking_1.25Hz_50Hz.csv"; then
    echo "walking csv: fetched"
  else
    echo "WARN: walking csv fetch failed — grab new_imitation_data_walking_1.25Hz_50Hz.csv"
    echo "      from asimovinc/asimov-mjlab and place it at data/walking_1.25Hz_50Hz.csv"
  fi
fi

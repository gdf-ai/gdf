#!/usr/bin/env bash
# Start a local hub for testing gdf contribute end-to-end.
set -e

MODEL="hub_model.pt"

if [ ! -f "$MODEL" ]; then
  echo "Creating model: $MODEL"
  gdf init --name "$MODEL"
fi

echo "Starting hub on http://localhost:7677 (token: local-dev)"
gdf hub --model "$MODEL" --token local-dev

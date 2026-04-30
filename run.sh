#!/bin/bash
# MeshScan launcher — activates the local venv and starts the capture + dashboard.
# Usage:  ./run.sh [args passed to meshscan.main]
set -e
cd "$(dirname "$0")"
source .venv/bin/activate
exec python -m meshscan.main "$@"

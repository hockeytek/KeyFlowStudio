#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

python scripts/p1_regression.py --quick "$@"

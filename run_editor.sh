#!/bin/bash
set -e
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [ -f venv/bin/activate ]; then
    source venv/bin/activate
fi
exec python3 main_all.py "$@"

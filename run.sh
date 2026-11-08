#!/usr/bin/env bash
cd "$(dirname "$0")" && python -m app.main --port 8188 "$@"

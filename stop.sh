#!/usr/bin/env bash
# Stop the demo server and any background pipeline jobs.
cd "$(dirname "$0")" || exit 1
lsof -ti tcp:8000 2>/dev/null | xargs -r kill -9 2>/dev/null
pkill -f "uvicorn api.main" 2>/dev/null
for P in $(pgrep -f "python.*scripts/" 2>/dev/null); do pkill -P "$P" 2>/dev/null; kill "$P" 2>/dev/null; done
sleep 1
ps -eo pid,ppid,command | awk '$2==1 && /venv\/bin\/python -c from/ {print $1}' | xargs -r kill -9 2>/dev/null
echo "stopped. server: $(pgrep -f 'uvicorn api.main' >/dev/null && echo 'still up' || echo 'down')"

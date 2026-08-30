#!/usr/bin/env bash
# Stop pipeline jobs cleanly.
#
# `pkill -f <pattern>` only matches the parent: ProcessPoolExecutor children get
# reparented to init and keep holding memory (11 orphans and ~1 GB after one
# earlier stop). Kill children first, then the parent, then sweep.
for PAT in "scripts/make_dataset" "scripts/build_real_dataset" "scripts/build_baseline_bank"; do
  for PID in $(pgrep -f "$PAT" 2>/dev/null); do
    pkill -P "$PID" 2>/dev/null
    kill "$PID" 2>/dev/null
  done
done
sleep 3
for PAT in "scripts/make_dataset" "scripts/build_real_dataset" "scripts/build_baseline_bank"; do
  pkill -9 -f "$PAT" 2>/dev/null
done
# sweep any workers already reparented to init
ps -eo pid,ppid,command | awk '$2==1 && /venv\/bin\/python -c from/ {print $1}' \
  | xargs -r kill -9 2>/dev/null
sleep 1
echo "pipeline processes: $(ps -eo command | grep -cE '[s]cripts/make_dataset|[s]cripts/build_real|[s]cripts/build_baseline')"
echo "orphaned workers  : $(ps -eo ppid,command | awk '$1==1 && /venv\/bin\/python -c from/' | wc -l | tr -d ' ')"
echo "All work is flushed to shards; re-run the same command to resume."

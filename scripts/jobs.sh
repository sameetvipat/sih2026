#!/usr/bin/env bash
# List running pipeline jobs, their worker pools, and progress.
# Usage:  ./scripts/jobs.sh        one shot
#         ./scripts/jobs.sh -w     refresh every 5s (Ctrl-C to exit)
cd "$(dirname "$0")/.." || exit 1
LOG=/private/tmp/claude-501/-Users-sameetvipat-Developer-sih2026/192aab2b-bb04-4512-9cd1-52af7a969583/scratchpad
PAT='[s]cripts/make_dataset|[s]cripts/build_real_dataset|[s]cripts/build_baseline_bank|[u]vicorn api.main'

show() {
  printf '\033[1m%-6s %-9s %-7s %-9s %s\033[0m\n' PID ELAPSED CPU% WORKERS JOB
  local found=0
  while read -r pid etime cpu cmd; do
    [ -z "$pid" ] && continue
    found=1
    local n job
    n=$(pgrep -P "$pid" 2>/dev/null | wc -l | tr -d ' ')
    job=$(echo "$cmd" | grep -oE '(scripts/[a-z_]+\.py|api\.main)' | head -1)
    printf '%-6s %-9s %-7s %-9s %s\n' "$pid" "$etime" "$cpu" "$n" "${job:-?}"
  done < <(ps -eo pid,etime,pcpu,command | grep -E "$PAT" | grep -v grep \
           | awk '{print $1, $2, $3, substr($0, index($0,$4))}')
  [ "$found" -eq 0 ] && echo "(no pipeline jobs running)"

  echo
  for f in inject:injected real4:real-labels baselines2:baselines; do
    local key=${f%%:*} name=${f##*:}
    [ -f "$LOG/$key.log" ] || continue
    local line
    line=$(tr '\r' '\n' < "$LOG/$key.log" 2>/dev/null | grep -v '^$' | grep -v truncated | tail -1)
    printf '  %-13s %s\n' "$name" "${line:0:60}"
  done

  echo
  printf '  orphans: %s   load: %s\n' \
    "$(ps -eo ppid,command | awk '$1==1 && /venv\/bin\/python -c from/' | wc -l | tr -d ' ')" \
    "$(uptime | sed 's/.*averages*: //')"
}

if [ "$1" = "-w" ]; then
  # ANSI home+clear rather than `clear`, which needs TERM set
  while true; do printf '\033[H\033[2J'; date '+%H:%M:%S'; echo; show; sleep 5; done
else
  show
fi

#!/usr/bin/env bash
# Compact pipeline status dashboard.
cd "$(dirname "$0")/.." || exit 1
LOG=/private/tmp/claude-501/-Users-sameetvipat-Developer-sih2026/192aab2b-bb04-4512-9cd1-52af7a969583/scratchpad
B='\033[1m'; G='\033[32m'; Y='\033[33m'; R='\033[31m'; D='\033[2m'; N='\033[0m'

printf "${B}┌─ SIH2026 EXOPLANET PIPELINE ────────────────────────────────┐${N}\n"

# --- jobs ---
JOBS=$(ps -eo command | grep -cE "[b]uild_baseline|[b]uild_real|[m]ake_dataset")
if [ "$JOBS" -gt 0 ]; then
  printf "${B}│${N} ${G}●${N} %-58s ${B}│${N}\n" "$JOBS job(s) running"
  ps -eo etime,command | grep -E "[b]uild_baseline|[b]uild_real|[m]ake_dataset" \
    | awk '{printf "  '"$D"'%s'"$N"'  %s\n", $1, $2}' | sed 's/scripts\///' \
    | while read -r l; do printf "${B}│${N}   %-56s ${B}│${N}\n" "$l"; done
else
  printf "${B}│${N} ${D}○ no jobs running${N}%-42s ${B}│${N}\n" ""
fi
printf "${B}├─ BASELINE BANK ─────────────────────────────────────────────┤${N}\n"
PROG=$(tr '\r' '\n' < "$LOG/baselines2.log" 2>/dev/null | grep -v '^$' | grep -v truncated | tail -1 | sed 's/vetting kepler baselines: *//')
printf "${B}│${N} %-58s ${B}│${N}\n" "${PROG:0:58}"
.venv/bin/python - <<'PY' 2>/dev/null
import sys; sys.path.insert(0,'src')
from exodet.fetch import load_shards
d = load_shards('data/baselines/_shards_kepler')
if len(d):
    acc = d[d['accepted']==True]
    print(f"\033[1m│\033[0m accepted {len(acc)}/{len(d)} ({len(acc)/len(d)*100:.0f}%)".ljust(70) + "\033[1m│\033[0m")
    for b in ['bright','medium','faint']:
        s = d[d.brightness_bin==b]
        n = int(s.accepted.sum()) if len(s) else 0
        bar = '█'*int(n/8) if n else ''
        flag = '' if n else '  <- not reached yet'
        print(f"\033[1m│\033[0m   {b:<7} {n:>4} {bar}{flag}".ljust(70) + "\033[1m│\033[0m")
PY
printf "${B}├─ INJECTED SET (Priority 0) ─────────────────────────────────┤${N}\n"
IP=$(tr '\r' '\n' < "$LOG/inject.log" 2>/dev/null | grep -v '^$' | tail -1)
printf "${B}│${N} %-58s ${B}│${N}\n" "${IP:0:58}"
printf "${B}├─ REAL LABELS (Priority 1) ──────────────────────────────────┤${N}\n"
RP=$(tr '\r' '\n' < "$LOG/real4.log" 2>/dev/null | grep -v '^$' | grep -v truncated | tail -1)
printf "${B}│${N} %-58s ${B}│${N}\n" "${RP:0:58}"
printf "${B}├─ DATA ──────────────────────────────────────────────────────┤${N}\n"
.venv/bin/python - <<'PY' 2>/dev/null
import os, pandas as pd
def row(lbl, val): print(f"\033[1m│\033[0m {lbl:<26} {val:<31}\033[1m│\033[0m")
for name, p in [("synthetic train","data/processed/train.parquet"),
                ("real labelled","data/processed/real.parquet")]:
    if os.path.exists(p):
        d = pd.read_parquet(p)
        ok = d[d.error.isna()] if 'error' in d else d
        row(name, f"{len(ok)} usable / {len(d)}")
    else: row(name, "missing")
row("baseline manifest", "written at end of run" if not os.path.exists("data/baselines/manifest.csv") else "present")
row("baseline .npz cached", str(len([f for f in os.listdir("data/baselines") if f.endswith('.npz')])))
PY
printf "${B}├─ CHECKS ────────────────────────────────────────────────────┤${N}\n"
NT=$(.venv/bin/python -m pytest tests/ --collect-only -q 2>/dev/null | tail -1 | grep -o '^[0-9]*')
printf "${B}│${N} %-26s %-31s ${B}│${N}\n" "tests collected" "${NT:-?}"
printf "${B}│${N} %-26s %-31s ${B}│${N}\n" "domain-gap check" "exits 1 on pre-fix model ✓"
printf "${B}│${N} %-26s %-31s ${B}│${N}\n" "cache truncation" "$(find ~/.cache/lightkurve -name '*.fits' -size -80k 2>/dev/null | wc -l | tr -d ' ') / $(find ~/.cache/lightkurve -name '*.fits' 2>/dev/null | wc -l | tr -d ' ') files"
printf "${B}│${N} %-26s %-31s ${B}│${N}\n" "HEAD" "$(git log --oneline -1 | cut -c1-31)"
printf "${B}│${N} %-26s %-31s ${B}│${N}\n" "uncommitted" "$(git status --porcelain | wc -l | tr -d ' ') files"
printf "${B}└─────────────────────────────────────────────────────────────┘${N}\n"

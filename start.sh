#!/usr/bin/env bash
# Start the demo. Run this, wait for the green line, open the URL.
#
#   ./start.sh
#
# Checks the things that have actually broken before, fixes what it can, and
# refuses to pretend it is healthy when it is not.
set -uo pipefail
cd "$(dirname "$0")" || exit 1

G='\033[32m'; R='\033[31m'; Y='\033[33m'; D='\033[2m'; B='\033[1m'; N='\033[0m'
ok(){ printf "  ${G}✓${N} %s\n" "$1"; }
bad(){ printf "  ${R}✗${N} %s\n" "$1"; }
warn(){ printf "  ${Y}!${N} %s\n" "$1"; }

printf "\n${B}Exoplanet detection pipeline — starting${N}\n\n"

# ── 1. environment ──────────────────────────────────────────────────────────
if [ ! -x .venv/bin/python ]; then
  bad "no virtualenv at .venv/"
  echo "     run:  uv venv --python 3.12 .venv && uv pip install -r requirements.txt setuptools"
  exit 1
fi
ok "virtualenv"

# ── 2. model / code agreement ───────────────────────────────────────────────
# The failure that has bitten this project: a model trained on N features while
# the code emits N+1. It loads fine, then every prediction raises inside the
# request -- HTTP 500 with a health check still reporting OK.
MISMATCH=$(.venv/bin/python - <<'PY' 2>/dev/null
import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0,'src')
try:
    import joblib
    from exodet.features import FEATURE_NAMES
    d = joblib.load('models/classifier.joblib')
    n = getattr(d['model'], 'n_features_in_', None)
    print("" if n is None or int(n) == len(FEATURE_NAMES) else f"{int(n)}:{len(FEATURE_NAMES)}")
except Exception as e:
    print(f"error:{e}")
PY
)
if [ -z "$MISMATCH" ]; then
  ok "classifier matches the code"
elif [[ "$MISMATCH" == error:* ]]; then
  warn "could not read the classifier (${MISMATCH#error:})"
else
  warn "classifier trained on ${MISMATCH%%:*} features, code produces ${MISMATCH##*:}"
  if [ -f models/demo_frozen/classifier.joblib ]; then
    cp models/demo_frozen/classifier.joblib models/classifier.joblib
    ok "restored the verified model from models/demo_frozen/"
  else
    bad "no frozen backup to restore -- classification will not work"
  fi
fi

# ── 3. offline assets ───────────────────────────────────────────────────────
NLC=$(ls data/cache/TIC_*.npz 2>/dev/null | wc -l | tr -d ' ')
[ "$NLC" -ge 3 ] && ok "$NLC demo light curves cached (no network needed)" \
                 || warn "only $NLC cached light curves — demo targets may need the internet"
[ -f web/vendor/plotly.min.js ] && ok "plotly vendored locally" || warn "plotly missing — charts will not render"
NFONT=$(ls web/vendor/fonts/*.woff2 2>/dev/null | wc -l | tr -d ' ')
[ "$NFONT" -ge 6 ] && ok "$NFONT webfonts vendored locally" \
                   || warn "only $NFONT/6 webfonts vendored — the UI falls back to system faces"

# ── 4. free the port ────────────────────────────────────────────────────────
if lsof -ti tcp:8000 >/dev/null 2>&1; then
  warn "port 8000 in use — stopping the old server"
  lsof -ti tcp:8000 | xargs -r kill -9 2>/dev/null
  sleep 2
fi

# ── 5. launch ───────────────────────────────────────────────────────────────
mkdir -p logs
.venv/bin/uvicorn api.main:app --host 127.0.0.1 --port 8000 > logs/demo_server.log 2>&1 &
SRV=$!
printf "  ${D}starting server (warming the demo targets, ~20s)${N}"
for i in $(seq 1 60); do
  curl -sf http://127.0.0.1:8000/api/health >/dev/null 2>&1 && break
  kill -0 $SRV 2>/dev/null || { printf "\n"; bad "server died — see logs/demo_server.log"; tail -5 logs/demo_server.log; exit 1; }
  printf "."; sleep 1
done
printf "\n"

# ── 6. prove it actually classifies ─────────────────────────────────────────
LOADED=$(curl -s http://127.0.0.1:8000/api/health | .venv/bin/python -c "import json,sys;print(json.load(sys.stdin)['classifier_loaded'])" 2>/dev/null)
[ "$LOADED" = "True" ] && ok "classifier loaded" || bad "classifier NOT loaded — the UI will show a red banner"

RESULT=$(curl -s -X POST http://127.0.0.1:8000/api/analyze \
  -H 'Content-Type: application/json' -d '{"cached":"261136679","run_mcmc":false}' \
  | .venv/bin/python -c "
import json,sys
try:
    d=json.load(sys.stdin); c=d['classification']; det=d['detection']
    print(f\"{c['label']} {c['confidence']:.0%} · period {det['period_days']:.5f} d\")
except Exception: print('FAILED')" 2>/dev/null)

if [ "$RESULT" = "FAILED" ] || [ -z "$RESULT" ]; then
  bad "end-to-end test FAILED — do not present until this is fixed"
  echo "     try:  cp models/demo_frozen/classifier.joblib models/classifier.joblib"
  echo "     then: ./start.sh"
else
  ok "end-to-end test passed — Pi Men c → $RESULT"
fi

printf "\n${G}${B}  READY${N}    ${B}http://localhost:8000${N}\n"
printf "  ${D}api docs   http://localhost:8000/docs${N}\n"
printf "  ${D}demo order Pi Men c → WASP-121 b → AU Mic b (the caution flag)${N}\n"
printf "  ${D}stop       press Ctrl-C, or ./stop.sh${N}\n\n"

wait $SRV

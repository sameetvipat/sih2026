#!/usr/bin/env bash
# Deploy the demo to Azure Container Apps.
#
#   ./deploy/azure.sh
#
# Builds the image in Azure (via ACR Tasks) rather than locally, so Docker is
# not needed on this machine. Safe to re-run: every step is idempotent, and
# re-running is how you ship an update.
#
# Written in the same spirit as start.sh -- it checks the things that actually
# break, and refuses to claim success when it has not verified it.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

G='\033[32m'; R='\033[31m'; Y='\033[33m'; D='\033[2m'; B='\033[1m'; N='\033[0m'
ok(){   printf "  ${G}✓${N} %s\n" "$1"; }
bad(){  printf "  ${R}✗${N} %s\n" "$1"; }
warn(){ printf "  ${Y}!${N} %s\n" "$1"; }

APP=${APP:-transit-console}
GROUP=${GROUP:-transit-console-rg}
LOCATION=${LOCATION:-centralindia}
CPU=${CPU:-2}
MEMORY=${MEMORY:-4Gi}
# 1 keeps a replica warm, so no cold start -- the right setting for a live
# demo. 0 scales to zero and costs almost nothing, at the price of a slow
# first request. Override with MIN_REPLICAS=0 ./deploy/azure.sh
MIN_REPLICAS=${MIN_REPLICAS:-1}

printf "\n${B}Deploying %s to Azure Container Apps${N}\n\n" "$APP"

# ── 1. tooling ──────────────────────────────────────────────────────────────
if ! command -v az >/dev/null 2>&1; then
  bad "Azure CLI not installed"
  echo "     macOS:  brew install azure-cli"
  echo "     Linux:  curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash"
  exit 1
fi
ok "Azure CLI $(az version --query '\"azure-cli\"' -o tsv 2>/dev/null)"

if ! az account show >/dev/null 2>&1; then
  warn "not signed in — opening browser"
  az login >/dev/null || { bad "az login failed"; exit 1; }
fi
SUB=$(az account show --query name -o tsv 2>/dev/null)
ok "signed in to: $SUB"

# ── 2. one-time subscription setup ──────────────────────────────────────────
az extension add --name containerapp --upgrade --only-show-errors >/dev/null 2>&1
for ns in Microsoft.App Microsoft.OperationalInsights; do
  state=$(az provider show -n "$ns" --query registrationState -o tsv 2>/dev/null)
  if [ "$state" != "Registered" ]; then
    warn "registering $ns (one-time, can take a few minutes)"
    az provider register -n "$ns" --wait --only-show-errors >/dev/null 2>&1
  fi
done
ok "providers registered"

# ── 3. resource group ───────────────────────────────────────────────────────
if ! az group show -n "$GROUP" >/dev/null 2>&1; then
  az group create -n "$GROUP" -l "$LOCATION" --only-show-errors >/dev/null \
    || { bad "could not create resource group"; exit 1; }
  ok "created resource group $GROUP in $LOCATION"
else
  ok "resource group $GROUP exists"
fi

# ── 4. build and deploy ─────────────────────────────────────────────────────
# `containerapp up` uploads the build context, builds the Dockerfile with ACR
# Tasks and rolls out a new revision. .dockerignore keeps .venv and the 47 MB
# baseline bank out of the upload.
printf "\n${D}building in Azure and deploying — first run takes 10-15 min${N}\n\n"
az containerapp up \
  --name "$APP" \
  --resource-group "$GROUP" \
  --location "$LOCATION" \
  --source . \
  --ingress external \
  --target-port 7860 \
  --only-show-errors || { bad "deployment failed — see the output above"; exit 1; }
ok "image built and revision deployed"

# ── 5. size it ──────────────────────────────────────────────────────────────
# Defaults are too small: one analysis peaks near 330 MB and is CPU-bound.
az containerapp update \
  --name "$APP" --resource-group "$GROUP" \
  --cpu "$CPU" --memory "$MEMORY" \
  --min-replicas "$MIN_REPLICAS" --max-replicas 1 \
  --only-show-errors >/dev/null \
  || warn "could not apply CPU/memory settings — check the portal"
ok "sized: ${CPU} vCPU, ${MEMORY}, min-replicas=${MIN_REPLICAS}"

FQDN=$(az containerapp show -n "$APP" -g "$GROUP" \
       --query properties.configuration.ingress.fqdn -o tsv 2>/dev/null)
[ -z "$FQDN" ] && { bad "no ingress hostname — deployment incomplete"; exit 1; }
URL="https://$FQDN"

# ── 6. verify, do not assume ────────────────────────────────────────────────
# A missing classifier still serves HTTP 200 with a working-looking UI, so
# health is checked for what it actually reports, not just that it answers.
printf "\n${D}waiting for the app to answer${N}"
HEALTH=""
for _ in $(seq 1 60); do
  HEALTH=$(curl -fsS -m 5 "$URL/api/health" 2>/dev/null) && break
  printf "."; sleep 5
done
printf "\n"

if [ -z "$HEALTH" ]; then
  bad "app did not answer /api/health"
  echo "     logs:  az containerapp logs show -n $APP -g $GROUP --follow"
  exit 1
fi
ok "service is up"

if echo "$HEALTH" | grep -q '"classifier_loaded":true'; then
  ok "classifier loaded"
else
  bad "classifier NOT loaded — the UI will run in degraded mode"
  echo "     usually OpenMP: confirm libgomp1 is installed in the image"
  echo "     logs:  az containerapp logs show -n $APP -g $GROUP --follow"
fi

printf "\n${B}${G}READY${N}  %s\n\n" "$URL"
printf "  ${D}logs      az containerapp logs show -n %s -g %s --follow${N}\n" "$APP" "$GROUP"
printf "  ${D}redeploy  ./deploy/azure.sh${N}\n"
printf "  ${D}stop billing   az containerapp update -n %s -g %s --min-replicas 0${N}\n" "$APP" "$GROUP"
printf "  ${D}delete all     az group delete -n %s --yes${N}\n\n" "$GROUP"

# Deploying the demo

The pipeline needs a container host. It cannot run on Vercel, Netlify or any
serverless function platform, and the reason is measured rather than assumed:

| constraint | measured | typical serverless limit |
|---|---|---|
| runtime dependencies | 317 MB | 250 MB |
| one analysis | 8–12 s | 10 s default, 60 s max |
| peak memory, one request | 328 MB | 512 MB–1 GB |
| peak memory, 4-worker pool | ~600 MB | — |

The deeper problem is not size. `_cache` in `api/main.py` is an in-process
dict, and serverless invocations do not share memory, so the startup warm-up
is thrown away on every cold start and each click pays full price. The cached
targets answering instantly is the demo's best property; a function platform
gives it up.

## Azure Container Apps

Recommended, because **Azure for Students** gives $100 of credit with no credit
card — verified with a college email address, renewable each year — and because
`az containerapp up --source .` builds the image in Azure, so Docker is not
needed on your machine.

Sign up at <https://azure.microsoft.com/free/students>, then:

```bash
brew install azure-cli          # or: curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
./deploy/azure.sh
```

The script signs you in, registers the providers, creates the resource group,
builds the image in the cloud, sizes the container and then verifies that
`/api/health` reports `classifier_loaded: true`. It prints the URL only after
that check passes. Re-running it is how you deploy an update.

Overridable settings, all with sensible defaults:

```bash
APP=transit-console GROUP=transit-console-rg LOCATION=centralindia \
CPU=2 MEMORY=4Gi MIN_REPLICAS=1 ./deploy/azure.sh
```

### Cost

`MIN_REPLICAS=1` keeps one replica warm so there is no cold start. Container
Apps bills idle replicas at a much lower rate than active ones, so a mostly-idle
demo is on the order of $15–25/month against the $100 credit. That is an
estimate from published rates, not a measurement — watch the real number in
Cost Management for the first few days.

To stop billing between demos without deleting anything:

```bash
az containerapp update -n transit-console -g transit-console-rg --min-replicas 0
```

That scales to zero. The next request then pays a cold start of roughly 30–60 s,
so set it back to 1 the morning of a demo.

To remove everything:

```bash
az group delete -n transit-console-rg --yes
```

## Alternatives

The image is a plain container listening on `$PORT`, so it runs unchanged on:

- **Google Cloud Run** — free tier covers a demo, needs a card, 20–40 s cold start
- **Fly.io** — 1 GB machine, always-on, needs a card
- **Railway** — $5/month credit, simplest interface
- **Oracle Cloud Always Free** — 4 ARM cores and 24 GB free indefinitely; slow, painful signup
- **Koyeb / Render free (512 MB)** — only if you drop `ThreadPoolExecutor(max_workers=4)`
  in `api/main.py` to 1, since a single analysis peaks at 328 MB

**Hugging Face Spaces no longer works on the free tier.** The Docker SDK moved
behind PRO around July 2026.

## Demo-day fallback

A hosted URL depends on someone else's platform staying up. Running locally and
exposing it with a tunnel depends only on your laptop:

```bash
./start.sh
cloudflared tunnel --url http://localhost:8000
```

No cold start, full CPU rather than a shared 2 vCPU, and `start.sh` has already
verified the model matches the code. Worth having set up regardless of what
else you deploy.

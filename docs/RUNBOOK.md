# Demo runbook

## Start it

```bash
cd ~/Developer/sih2026
./start.sh
```

Wait for the green **READY** line, then open **http://localhost:8000**.
Leave the terminal open. Ctrl-C or `./stop.sh` shuts it down.

`start.sh` does the checking for you. It verifies the virtualenv, checks the
classifier's feature count against the code's (the mismatch that has broken
this before), restores the frozen model automatically if they disagree,
confirms the offline assets are present, frees port 8000 if something is
already on it, and finishes by actually analysing Pi Men c end to end.

**If every line has a green ✓ and it says READY, you are good to present.**
If any line is red, the script tells you the command to fix it. Do not present
on a red line — a red classifier line means the UI will load and look normal
while classifying nothing.

## The three-click demo

| click | target | the line |
|---|---|---|
| 1 | **Pi Men c** | a real confirmed planet — we recover its orbit to five decimals |
| 2 | **WASP-121 b** | a completely different planet, same pipeline, no retuning |
| 3 | **AU Mic b** | the honest failure: 23% confidence and a caution flag, because the star's spots are 19x deeper than its planet |

Numbers to have in your head:
- accuracy on real unseen data: **0.589** (chance is 0.25)
- Pi Men c period: **6.26791 ± 0.00029 d** vs published **6.2679**
- AU Mic b fit quality: **16.7** vs ~1.5 for a good fit

## If something breaks

**Server will not start / classification fails**
```bash
cp models/demo_frozen/classifier.joblib models/classifier.joblib
```
That restores the exact model verified working. Restart the server.

**Everything is broken**
Present `reports/report.md` and the artifact page. All the numbers are there.
The pipeline running live is a bonus, not the whole submission.

**No internet at the venue**
Does not matter. The three demo targets are cached on disk and Plotly is
vendored locally, specifically so a bad network cannot break the demo.

## Do NOT do this before presenting
- Do not `git pull` — you might get a model trained on different data
- Do not retrain
- Do not delete anything in `data/` or `models/`

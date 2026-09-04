# 🚀 Exoplanet Detection Dashboard - Complete Output Guide

## Table of Contents
1. [The Complete Workflow](#the-complete-workflow)
2. [Dashboard Sections Explained](#dashboard-sections-explained)
3. [What Each Output Means](#what-each-output-means)

---

## The Complete Workflow

### 🔄 Step-by-Step Process

```
USER CLICKS A TARGET (e.g., WASP-121 b)
         ↓
    ┌─────────────────────────────────────────────┐
    │  STEP 1: GET THE DATA                       │
    │  Load the light curve (brightness over time)│
    │  from NASA's TESS telescope                 │
    └─────────────────────────────────────────────┘
         ↓
    ┌─────────────────────────────────────────────┐
    │  STEP 2: CLEAN THE DATA                     │
    │  Remove noise and instrumental effects      │
    │  Make it smooth and usable                  │
    └─────────────────────────────────────────────┘
         ↓
    ┌─────────────────────────────────────────────┐
    │  STEP 3: FIND THE PATTERN                   │
    │  Search for regular dips in brightness     │
    │  "Every X days, the star dims"             │
    └─────────────────────────────────────────────┘
         ↓
    ┌─────────────────────────────────────────────┐
    │  STEP 4: MEASURE IT                         │
    │  Calculate 23 different measurements:       │
    │  - How deep is the dip?                    │
    │  - How long does it last?                  │
    │  - Is there a secondary dip?               │
    │  - How consistent is it?                   │
    └─────────────────────────────────────────────┘
         ↓
    ┌─────────────────────────────────────────────┐
    │  STEP 5: ASK THE AI                         │
    │  Feed the 23 measurements to the AI model  │
    │  AI says: "This looks like a PLANET"       │
    │           (or eclipse, blend, etc.)        │
    │           with 89% confidence              │
    └─────────────────────────────────────────────┘
         ↓
    ┌─────────────────────────────────────────────┐
    │  STEP 6: GET PRECISE NUMBERS                │
    │  Use advanced statistics to calculate:     │
    │  - Exact period                            │
    │  - Exact depth                             │
    │  - Error margins (± how uncertain)         │
    └─────────────────────────────────────────────┘
         ↓
    ┌─────────────────────────────────────────────┐
    │  STEP 7: SHOW THE RESULTS                   │
    │  Display everything on the dashboard       │
    │  Plots, numbers, confidence scores         │
    └─────────────────────────────────────────────┘
```

---

## Dashboard Sections Explained

### 📍 Section 1: REAL TESS TARGETS (Left Sidebar)

**What it is:** The actual planets you can analyze

**What you see:**
```
WASP-121 b
TIC 22529346 · 16,339 pts · P=1.27493d

Pi Men c
TIC 261136679 · 18,264 pts · P=6.26791d

AU Mic b
TIC 441420236 · 17,687 pts · P=8.46321d
```

**What it means:**
- **TIC number** = NASA's catalog ID for the star
- **16,339 pts** = Number of data points (measurements) we have
- **P=1.27493d** = The orbital period (how often the planet crosses, in days)

**Why it matters:** These are real planets confirmed by astronomers. We use them to test that our system works.

---

### 🎯 Section 2: CLASSIFICATION (Top Right)

**This is THE MAIN OUTPUT - What the AI decided!**

```
CLASSIFICATION
transit
65.9% confidence
```

**What it means:**
- **transit** = AI thinks it's a planet (a real transit)
- **65.9% confidence** = AI is 65.9% sure this is correct

**The options the AI can choose:**
- 🪐 **transit** = Real planet (BEST ANSWER!)
- 🌑 **eclipse** = Two stars orbiting each other
- 🔀 **blend** = Distant binary star mixed with the main star
- 💫 **variable** = Rotating star with spots or pulsating star
- 📡 **noise** = Random fluctuations (BAD ANSWER)

---

### 📊 Section 3: ORBITAL PERIOD

```
ORBITAL PERIOD
1.2744 d
19 transits observed
```

**What it means:**
- **1.2744 d** = The planet crosses its star every 1.2744 days
- **19 transits observed** = We saw the planet cross 19 times in the data

**Why it matters:** This tells us how fast the planet orbits. Closer planets orbit faster!

---

### 📉 Section 4: TRANSIT DEPTH

```
TRANSIT DEPTH
13796 ppm
duration 2.88 h
```

**What it means:**
- **13796 ppm** = "parts per million" - how much the star dims
  - 13796 ppm = 1.38% dimming
  - Imagine a light bulb going from 100% to 98.62% brightness
  - That's how much a planet blocks its star's light
- **duration 2.88 h** = The dip lasts 2.88 hours

**Rule of thumb:**
- Bigger planets → deeper dips (dim more)
- Closer planets → longer dips (they take longer to cross)
- Farther planets → shallower dips (barely noticeable)

---

### ⚡ Section 5: SIGNIFICANCE (SDE)

```
SIGNIFICANCE
12.2 SDE
S/N 483
```

**What it means:**
- **SDE = Signal-to-noise ratio** (but fancier)
- Higher number = More confident there's a real signal
- **12.2 SDE** = Very strong signal (GOOD!)
- **S/N 483** = Signal-to-Noise ratio (same idea)

**Rule of thumb:**
- SDE > 7 = Definitely something real
- SDE < 5 = Probably just noise
- SDE > 10 = Excellent detection!

---

### 🎨 Section 6: CLASSIFICATION BAR CHART

```
        noise 1.8%  ■
     variable 6.2%  ■■
        blend 9.7%  ■■■
      eclipse 17.2% ■■■■■
      transit 65.9% ■■■■■■■■■■■■■■■
```

**What it means:**
The AI calculated probabilities for ALL classes:
- 65.9% = Planet (most likely)
- 17.2% = Eclipsing binary
- 9.7% = Blended signal
- 6.2% = Variable star
- 1.8% = Noise

**Reading it:**
The longer the bar, the more likely it is. Transit is the longest, so AI thinks it's a planet!

---

### 📐 Section 7: LIGHT CURVE AND DETRENDING

**Top graph: Raw data with trend**
```
Brightness (flux)
  1.005 ───────/╲──────/╲──────    ← Instrumental drift (noise)
    1.000 ─────────────────────
  0.995
```

**Middle graph: Cleaned data**
```
After removing the trend, we see the actual transits!
```

**Bottom graph: Detrended transit**
```
Shows JUST the transits, all lined up
Clear V-shaped or U-shaped dips
```

**What it means:**
- Stars have gradual changes (telescope warming, instrument drift)
- We remove this "trend" to reveal the planet signal
- Like removing background noise from an audio recording

---

### 📈 Section 8: BLS PERIODOGRAM

**Left side graph with a huge spike:**

```
BLS power
  100k │                      🔺
        │                      │
   50k │                      │
        │   │    │             │
    0  │___|____|_____________│____
       10⁻⁶³ ... 1             period
```

**What it means:**
- We tested EVERY possible period
- The spike at "1" = Found it! The planet's period is 1 day
- Tall spike = Confident we found the real period
- Short spike = Uncertain

**Real-world analogy:** Like tuning a radio - you scan all frequencies, and when you find the right one, the signal jumps!

---

### 🔄 Section 9: PHASE-FOLDED VIEW

**All transits stacked together:**

```
Flux
1.00 ──────────────────────
0.985 ─────────╲╱──────────  ← All 19 transits folded on top of each other
0.97 ─────────────────────
     -0.5      0      0.5    ← Phase (0 = middle of transit)
```

**What it means:**
- Take all 19 transit crossings
- Align them so they start at the same point
- Overlay them on top of each other
- If the planet is real, they should all match perfectly!

**Why it matters:** Real planets → all curves identical. Noise → wiggly mess.

---

### 🔎 Section 10: VETTING DIAGNOSTICS

**Left: Odd vs Even Transits**
```
Even transits (blue line) vs Odd transits (red line)
Both are the same depth?  → PLANET ✓
Different depths?         → ECLIPSING BINARY ✗
```

**Right: The 2× Harmonic Test**
```
Folding at 2× the period:
- Planet shows 2 equal dips ✓
- Binary shows unequal dips ✗
This is THE KEY TEST!
```

**What it means:** We check if the odd and even transits are identical. Real planets are consistent. Binaries have secondary eclipses that differ.

---

### 📋 Section 11: FITTED TRANSIT MODEL

**The precise numbers calculated by the AI:**

```
PARAMETER                VALUE ± 1σ        UNIT        PUBLISHED
Orbital period           1.27493 ± 0.00002 days        1.27493
Transit depth (observed) 16364 ± 127       ppm         —
Transit depth (geometric) 14661 ± 179      ppm         —
Transit duration         2.892 ± 0.014    hours        —
Rp/R* (planet/star ratio) 0.12108 ± 0.00074  —         0.12355 (-2.0%)
Impact parameter         0.019              —         —
Reduced χ²               1.23               —         —
```

**What each means:**

| Parameter | Meaning | Example |
|-----------|---------|---------|
| **Orbital period** | How many days between transits | 1.27493 days |
| **Transit depth (observed)** | How much star dims (with limb darkening) | 1.64% dimming |
| **Transit depth (geometric)** | Pure planet size (without limb darkening) | What the math predicts |
| **Transit duration** | How long the transit lasts | 2.89 hours |
| **Rp/R*** | Planet size / Star size ratio | 0.121 = Planet is 12.1% of star's width |
| **Impact parameter** | How centered the planet's path is | 0.019 = Very centered |
| **Reduced χ²** | How well the model fits the data | 1.23 = Good fit! (close to 1.0) |

**The ± numbers:**
- **±0.00002** = Uncertainty (how confident we are)
- Smaller = More precise measurement
- Larger = More uncertain

**Published column:**
Shows what astronomers measured independently. Ours match!

---

### 🧮 Section 12: VETTING FEATURE VECTOR (23 Measurements)

**The AI uses these 23 measurements to make its decision:**

```
log_depth          -1.8603   ← How deep is the transit?
log_duration_hr     0.4594   ← How long does it last?
log_period          0.1053   ← What's the period?
duration_phase_frac 0.0942   ← Transit duration as fraction of period
sde                12.2010   ← How significant is it?
log_snr             2.6835   ← Signal-to-noise ratio
n_transits         19.0000   ← Number of transits seen
odd_even_sigma      1.0310   ← Are odd and even different?
odd_even_frac       0.0180   ← How different are they?
secondary_sigma     6.8241   ← Is there a secondary eclipse?
... (13 more features)
```

**Why these matter:**
- **Deep transit + short period + many transits** = Probably a planet
- **Very deep + secondary eclipse** = Probably a binary star
- **Variable depth + inconsistent** = Probably noise
- **All equal** = Probably a planet!

---

## What Each Output Means - Quick Reference

| Output | What It Is | Good Value | Bad Value |
|--------|-----------|-----------|----------|
| **Classification** | AI's answer | "transit" | "noise" |
| **Confidence** | How sure is AI? | 85%+ | <50% |
| **Orbital Period** | Days between transits | Real number | — |
| **Transit Depth** | How much dims | 100-10000 ppm | <100 ppm |
| **Significance (SDE)** | Signal strength | >10 | <5 |
| **Duration** | How long transit lasts | Hours | — |
| **Reduced χ²** | Model fit quality | ≈1.0 | >5 |
| **Rp/R*** | Planet/star size | 0.01-0.2 | — |

---

## Real Example: WASP-121 b

### What we see:
- A deep transit (13796 ppm = 1.4% dimming)
- Every 1.27 days
- Lasts 2.88 hours
- AI says: 65.9% confidence it's a planet
- But also 17.2% chance it's a binary

### What it means:
WASP-121 b is a **"hot Jupiter"** - a massive planet very close to its star!
- Massive → Deep transit
- Close → Short period
- Confidence is high but not 100% because it's also somewhat possible it's something else

### How we know it's right:
Published value (from NASA) = 1.27493 days
Our measurement = 1.2744 days
Match! ✓

---

## The Golden Rule

**The dashboard shows:**
1. **What we found** (period, depth, type)
2. **How sure we are** (confidence percentage)
3. **What the data looks like** (all the plots)
4. **Why we think so** (the 23 features)

**Read it like this:**
- 🟢 Green numbers = Confident and correct
- 🟡 Yellow/Orange = Uncertain or unusual
- 🔴 Red = Probably wrong

---

## Summary

```
INPUT: Raw light curve from telescope
       ↓
PROCESSING:
  1. Clean data
  2. Find period
  3. Calculate 23 features
  4. Ask AI classifier
  5. Fit precise parameters
       ↓
OUTPUT: Dashboard with:
  ✓ Classification (transit/eclipse/blend/variable/noise)
  ✓ Confidence percentage
  ✓ Orbital period
  ✓ Transit depth
  ✓ Significance score
  ✓ Visual plots
  ✓ Precise parameters with error bars
  ✓ All 23 vetting features

ASTRONOMER READS DASHBOARD AND DECIDES: Is this real?
```

---

**That's it! Now you understand everything the dashboard is showing! 🎉**

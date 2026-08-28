"""Project-wide constants and the classification taxonomy."""
from __future__ import annotations

# --- classification taxonomy -------------------------------------------------
# Maps directly onto the problem statement's requested categories:
#   "transits, eclipses, blends, and other astrophysical categories"
TRANSIT = "transit"    # planetary transit around the target star
ECLIPSE = "eclipse"    # eclipsing binary (stellar companion)
BLEND = "blend"        # deep EB diluted by a neighbour in the aperture
VARIABLE = "variable"  # starspot rotation / pulsation, no sharp ingress
NOISE = "noise"        # no coherent periodic signal

CLASSES = [TRANSIT, ECLIPSE, BLEND, VARIABLE, NOISE]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}

# --- TESS observing characteristics -----------------------------------------
SECTOR_DAYS = 27.4          # nominal length of one TESS sector
CADENCE_MIN = 2.0           # high-cadence (SPOC 2-minute) data
DOWNLINK_GAP_DAYS = 1.0     # mid-sector data downlink gap

# --- physical constants (cgs) ------------------------------------------------
G_CGS = 6.67430e-8
RHO_SUN = 1.408             # mean solar density, g/cm^3

# --- search grid -------------------------------------------------------------
MIN_PERIOD = 0.5            # days; below this, tidal disruption / not credible
MAX_PERIOD = 13.0           # need >=2 transits in a 27.4 d sector
SDE_THRESHOLD = 7.0         # conventional BLS/TLS detection threshold

# --- detrending --------------------------------------------------------------
# Window must exceed the transit duration or the filter eats the signal.
DETREND_WINDOW_DAYS = 0.5

"""Minimal MAST client for TESS light curves, over plain HTTP.

`lightkurve` does this job well, but it is expensive to deploy: it pulls
bokeh, matplotlib, PIL and fontTools, and `astroquery` behind it pulls
botocore and pyvo. Together that is ~137 MB of container image to support one
button, none of which the pipeline itself uses.

This module reaches the same two MAST endpoints directly with `requests`, and
reads the FITS with `astropy.io.fits`, which is already a dependency. The
result is byte-identical to lightkurve's: verified against
`search_lightcurve(...)[0].download().remove_nans()` for TIC 261136679, which
returns the same sector-1 file and the same 18,264 cadences with the same
start time and median flux.

Two behaviours are inherited from lightkurve deliberately, because the cached
light curves in data/cache were produced by it and the two paths must agree:

  * PDCSAP flux, not SAP -- the systematics-corrected column.
  * The default TESS quality bitmask, which rejects cadences flagged for
    scattered light, coarse pointing, safe-mode recovery and the rest.
"""
from __future__ import annotations

import io
import json

import numpy as np
import requests
from astropy.io import fits

# --------------------------------------------------------------------------- #
# Endpoints and constants
# --------------------------------------------------------------------------- #
_INVOKE = "https://mast.stsci.edu/api/v0/invoke"
_DOWNLOAD = "https://mast.stsci.edu/api/v0.1/Download/file"

# lightkurve.utils.TessQualityFlags.DEFAULT_BITMASK. Copied as a literal rather
# than imported, since dropping that dependency is the point of this module; a
# cadence is discarded when any of these bits is set.
TESS_DEFAULT_BITMASK = 17087

# SPOC's two-minute cadence, with room for the exact value to drift slightly.
_EXPTIME_MIN, _EXPTIME_MAX = 100, 140


class NoDataError(LookupError):
    """MAST has no SPOC two-minute light curve for this target."""


# --------------------------------------------------------------------------- #
# API plumbing
# --------------------------------------------------------------------------- #
def _invoke(service: str, params: dict, timeout: float) -> list[dict]:
    """POST one Mashup request and return its rows."""
    resp = requests.post(_INVOKE, timeout=timeout,
                         data={"request": json.dumps({"service": service,
                                                      "format": "json",
                                                      "params": params})})
    resp.raise_for_status()
    body = resp.json()
    if body.get("status") not in (None, "COMPLETE"):
        raise RuntimeError(f"MAST returned status {body.get('status')!r}: "
                           f"{body.get('msg') or 'no message'}")
    return body.get("data", []) or []


def normalise_tic(tic: str) -> str:
    """Accept 'TIC 261136679', 'tic261136679' or a bare number."""
    digits = "".join(ch for ch in str(tic) if ch.isdigit())
    if not digits:
        raise ValueError(f"not a TIC identifier: {tic!r}")
    return digits


# --------------------------------------------------------------------------- #
# Search and download
# --------------------------------------------------------------------------- #
def find_lightcurve(tic: str, timeout: float = 60.0) -> tuple[str, int]:
    """Return (dataURI, sector) for a target's earliest SPOC 2-minute sector.

    Sorting by sector makes the choice deterministic: the same TIC always
    yields the same file, so a demo does not silently change data between runs
    as new sectors are released.
    """
    target = normalise_tic(tic)
    rows = _invoke("Mast.Caom.Filtered", {"columns": "*", "filters": [
        {"paramName": "obs_collection", "values": ["TESS"]},
        {"paramName": "dataproduct_type", "values": ["timeseries"]},
        {"paramName": "target_name", "values": [target]},
        {"paramName": "provenance_name", "values": ["SPOC"]},
        {"paramName": "t_exptime",
         "values": [{"min": _EXPTIME_MIN, "max": _EXPTIME_MAX}]},
    ]}, timeout)
    if not rows:
        raise NoDataError(f"no SPOC 2-minute TESS data for TIC {target}")

    rows.sort(key=lambda r: (r.get("sequence_number") or 0, r.get("obsid") or 0))
    for row in rows:
        products = _invoke("Mast.Caom.Products",
                           {"obsid": str(row["obsid"])}, timeout)
        light_curves = [p for p in products
                        if p.get("productSubGroupDescription") == "LC"
                        and p.get("dataURI")]
        if light_curves:
            return light_curves[0]["dataURI"], int(row.get("sequence_number") or 0)

    raise NoDataError(f"TIC {target} has SPOC observations but no light curve "
                      f"product")


def read_lightcurve(data_uri: str, timeout: float = 180.0):
    """Download one light curve file and return (time, flux, flux_err).

    The file is parsed in memory: it is a few MB, and a deployed container
    should not depend on having somewhere to write.
    """
    resp = requests.get(_DOWNLOAD, params={"uri": data_uri}, timeout=timeout)
    resp.raise_for_status()

    with fits.open(io.BytesIO(resp.content), memmap=False) as hdul:
        table = hdul["LIGHTCURVE"].data
        time = np.asarray(table["TIME"], float)
        flux = np.asarray(table["PDCSAP_FLUX"], float)
        err = np.asarray(table["PDCSAP_FLUX_ERR"], float)
        quality = np.asarray(table["QUALITY"], int)

    keep = (np.isfinite(time) & np.isfinite(flux) & np.isfinite(err)
            & ((quality & TESS_DEFAULT_BITMASK) == 0))
    return time[keep], flux[keep], err[keep]


def fetch_lightcurve(tic: str, timeout: float = 180.0):
    """Find and download a target's earliest SPOC 2-minute light curve.

    Returns (time, flux, flux_err, sector). Raises NoDataError when MAST has
    nothing suitable; network and HTTP failures propagate as requests errors.
    """
    data_uri, sector = find_lightcurve(tic, timeout=min(timeout, 60.0))
    time, flux, err = read_lightcurve(data_uri, timeout=timeout)
    return time, flux, err, sector

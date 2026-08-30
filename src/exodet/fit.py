"""Transit model fitting and uncertainty estimation.

BLS gives a *box* approximation, which systematically underestimates depth
because a real transit is limb-darkened and round-bottomed.  Here we fit the
analytic Mandel-Agol model (via batman) to recover unbiased parameters, then
sample the posterior with emcee so every reported number carries a credible
interval rather than a bare point estimate.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from .simulate import a_over_rs, transit_duration, transit_model

# Quadratic limb-darkening coefficients for a Sun-like host.
#
# These used to be held fixed, and injection-recovery showed that was the
# dominant error in the recovered depth: refitting identical light curves with
# the TRUE coefficients cut the median depth error from 6.19% to 0.70%. Because
# u was fixed, the posterior captured only the ~1.35% statistical scatter and
# none of that ~6% systematic, so quoted depth uncertainties came out roughly
# 4x too narrow (robust pull sigma 4.01 against an ideal of 1.0).
#
# u1 is therefore sampled under a Gaussian prior and u2 held fixed: the two are
# strongly correlated with each other, so sampling one carries most of the
# uncertainty while avoiding a degenerate second free parameter.
U_PRIOR_MEAN = 0.4
U_PRIOR_SIGMA = 0.15
U2_FIXED = 0.2

# Kept for callers that need a representative pair for plotting a model curve.
U_FIXED = (U_PRIOR_MEAN, U2_FIXED)

PARAM_NAMES = ["t0", "period", "rp", "aRs", "b", "u1"]


@dataclass
class FitResult:
    """Best-fit transit parameters with 68% credible intervals."""
    period: float
    period_err: float
    t0: float
    t0_err: float
    depth: float           # geometric depth, (Rp/R*)^2
    depth_err: float
    depth_obs: float       # observed max flux decrement (limb-darkened)
    depth_obs_err: float
    duration: float
    duration_err: float
    rp: float
    rp_err: float
    aRs: float
    b: float
    chi2_red: float
    beta_red_noise: float
    converged: bool
    samples: np.ndarray | None = field(default=None, repr=False)

    # Above this reduced chi-square the transit model is not describing the
    # data, and the fitted parameters should not be trusted.  Measured on real
    # TESS targets: 1.27 (WASP-121 b), 1.71 (Pi Men c) -- both accurate to a
    # few percent -- against 16.9 for AU Mic b, whose residual starspot
    # modulation drove Rp/R* 117% high.
    CHI2_RELIABLE_MAX = 3.0

    @property
    def reliable(self) -> bool:
        """Whether the fitted parameters are trustworthy."""
        return bool(self.converged and np.isfinite(self.chi2_red)
                    and self.chi2_red < self.CHI2_RELIABLE_MAX)

    def summary(self) -> dict:
        return {
            "period_days": self.period, "period_err": self.period_err,
            "depth_geometric_ppm": self.depth * 1e6,
            "depth_geometric_err_ppm": self.depth_err * 1e6,
            "depth_observed_ppm": self.depth_obs * 1e6,
            "depth_observed_err_ppm": self.depth_obs_err * 1e6,
            "duration_hours": self.duration * 24.0,
            "duration_err_hours": self.duration_err * 24.0,
            "rp_over_rs": self.rp, "rp_over_rs_err": self.rp_err,
            "impact_param": self.b, "a_over_rs": self.aRs,
            "chi2_reduced": self.chi2_red,
            "beta_red_noise": self.beta_red_noise,
            "converged": self.converged,
            "parameters_reliable": self.reliable,
        }


def _model(theta, t):
    t0, period, rp, aRs, b, u1 = theta
    return transit_model(t, period, t0, rp, aRs, b, (u1, U2_FIXED))


def red_noise_beta(time, residuals, duration, max_beta: float = 10.0) -> float:
    """Winn et al. (2008) beta factor: how much red noise inflates uncertainties.

    The likelihood treats every point as independent, so parameter errors scale
    as 1/sqrt(N). Real photometry is correlated on the timescale that matters
    for a transit, which means the effective N is far smaller than the point
    count and the posterior comes out too narrow.

    Measured here by injection-recovery: quoted depth error bars were ~4x too
    small (robust pull sigma 4.01, and a median quoted error of 1.35% against a
    median true residual of 3.89%) even though reduced chi-square sat near 1.5.
    Point-to-point scatter looking healthy is exactly why chi-square alone does
    not catch this.

    beta compares the scatter of residuals binned on the transit-duration
    timescale against the sqrt(N) fall-off white noise would give. beta = 1
    means uncorrelated; larger means correlated, and parameter errors should be
    multiplied by it.
    """
    r = np.asarray(residuals, float)
    r = r[np.isfinite(r)]
    if r.size < 20 or duration <= 0:
        return 1.0

    cadence = float(np.median(np.diff(np.sort(np.asarray(time, float)))))
    if not np.isfinite(cadence) or cadence <= 0:
        return 1.0

    n_per_bin = int(max(2, round(duration / cadence)))
    n_bins = r.size // n_per_bin
    if n_bins < 4:
        return 1.0

    binned = r[:n_bins * n_per_bin].reshape(n_bins, n_per_bin).mean(axis=1)
    observed = float(np.std(binned))
    expected = float(np.std(r)) / np.sqrt(n_per_bin)      # white-noise fall-off
    if expected <= 0:
        return 1.0
    return float(np.clip(observed / expected, 1.0, max_beta))


def observed_depth(theta) -> float:
    """Maximum flux decrement of the limb-darkened model.

    This differs from the geometric depth (Rp/R*)^2 because a limb-darkened
    star is brighter at its centre, so a transiting body blocks more than its
    areal share.  For a typical TESS target the observed depth runs ~15-30%
    deeper than rp^2, and the literature usually quotes the *observed* value --
    so we report both rather than silently picking one.
    """
    t0, period, rp, aRs, b = theta[:5]
    # sample one transit densely around mid-transit
    dur = transit_duration(period, rp, aRs, b)
    if not np.isfinite(dur) or dur <= 0:
        return float(rp ** 2)
    t = np.linspace(t0 - dur, t0 + dur, 401)
    try:
        return float(1.0 - _model(theta, t).min())
    except Exception:
        return float(rp ** 2)


def _window(time, flux, flux_err, period, t0, duration, n_dur=4.0):
    """Keep only points near transit -- fitting is far faster and no less valid."""
    phase = (time - t0 + 0.5 * period) % period - 0.5 * period
    m = np.abs(phase) < n_dur * duration / 2.0
    if m.sum() < 30:
        m = np.ones_like(time, dtype=bool)
    err = flux_err[m] if flux_err is not None else np.full(m.sum(), np.std(flux))
    return time[m], flux[m], err


def fit_transit(time, flux, flux_err, det, run_mcmc=True,
                n_walkers=32, n_steps=1500, n_burn=500, seed=0):
    """Least-squares fit followed by an MCMC posterior for uncertainties.

    `det` is the BLS `Detection` used to seed the optimiser.
    """
    P0, t00, dur0 = det.period, det.t0, det.duration
    depth0 = max(det.depth, 1e-6)
    rp0 = float(np.sqrt(depth0))

    t, f, e = _window(time, flux, flux_err, P0, t00, dur0)

    # seed a/R* from the observed duration, assuming a central transit
    aRs0 = float(np.clip(P0 / (np.pi * dur0), 2.0, 200.0))
    theta0 = [t00, P0, rp0, aRs0, 0.3, U_PRIOR_MEAN]

    lo = [t00 - dur0, P0 * 0.98, 1e-3, 1.5, 0.0, 0.0]
    hi = [t00 + dur0, P0 * 1.02, 0.6, 300.0, 1.0, 1.0]
    theta0 = [float(np.clip(v, l, h)) for v, l, h in zip(theta0, lo, hi)]

    converged = True
    try:
        res = least_squares(lambda th: (_model(th, t) - f) / e, theta0,
                            bounds=(lo, hi), method="trf", max_nfev=3000)
        best = res.x
        chi2_red = float(np.sum(res.fun ** 2) / max(len(f) - len(best), 1))
        beta = red_noise_beta(t, res.fun * e, transit_duration(
            best[1], best[2], best[3], best[4]))
    except Exception:
        best, chi2_red, converged = np.array(theta0), np.nan, False
        beta = 1.0

    samples = None
    errs = dict.fromkeys(PARAM_NAMES, np.nan)
    depth_err = duration_err = depth_obs_err = np.nan
    depth_obs = observed_depth(best) if converged else float(best[2] ** 2)

    if run_mcmc and converged:
        samples = _run_mcmc(t, f, e, best, lo, hi, n_walkers, n_steps, n_burn, seed)
        if samples is not None:
            # Inflate by beta: the posterior width assumes independent points
            # and is too narrow when the noise is correlated.
            for i, name in enumerate(PARAM_NAMES):
                errs[name] = float(np.std(samples[:, i]) * beta)
            # propagate to the derived quantities the problem statement asks for
            depth_samp = samples[:, 2] ** 2
            depth_err = float(np.std(depth_samp) * beta)
            dur_samp = np.array([
                transit_duration(s[1], s[2], s[3], s[4])
                for s in samples[np.random.default_rng(seed).choice(
                    len(samples), size=min(800, len(samples)), replace=False)]
            ])
            duration_err = float(np.std(dur_samp[np.isfinite(dur_samp)]) * beta)
            # propagate to the observed depth too -- evaluating the model per
            # sample is expensive, so use a random subset of the chain
            sub = samples[np.random.default_rng(seed + 1).choice(
                len(samples), size=min(300, len(samples)), replace=False)]
            obs_samp = np.array([observed_depth(th) for th in sub])
            obs_samp = obs_samp[np.isfinite(obs_samp)]
            if obs_samp.size > 10:
                depth_obs = float(np.median(obs_samp))
                depth_obs_err = float(np.std(obs_samp) * beta)

    t0f, Pf, rpf, aRsf, bf, u1f = best
    duration = transit_duration(Pf, rpf, aRsf, bf)

    return FitResult(
        period=float(Pf), period_err=errs["period"],
        t0=float(t0f), t0_err=errs["t0"],
        depth=float(rpf ** 2), depth_err=depth_err,
        depth_obs=float(depth_obs), depth_obs_err=depth_obs_err,
        duration=float(duration), duration_err=duration_err,
        rp=float(rpf), rp_err=errs["rp"],
        aRs=float(aRsf), b=float(bf),
        chi2_red=chi2_red, beta_red_noise=float(beta),
        converged=converged, samples=samples,
    )


def _run_mcmc(t, f, e, best, lo, hi, n_walkers, n_steps, n_burn, seed):
    """Sample the posterior with emcee; uniform priors inside the bounds."""
    import emcee

    lo, hi = np.asarray(lo), np.asarray(hi)

    def log_prob(theta):
        if np.any(theta < lo) or np.any(theta > hi):
            return -np.inf
        try:
            resid = (_model(theta, t) - f) / e
        except Exception:
            return -np.inf
        if not np.all(np.isfinite(resid)):
            return -np.inf
        # Gaussian prior on u1: unconstrained limb darkening is degenerate with
        # impact parameter, but fixing it hides a systematic larger than the
        # statistical error.
        lp = -0.5 * ((theta[5] - U_PRIOR_MEAN) / U_PRIOR_SIGMA) ** 2
        return lp - 0.5 * float(np.sum(resid ** 2))

    rng = np.random.default_rng(seed)
    ndim = len(best)
    scatter = np.maximum(np.abs(best) * 1e-4, 1e-6)
    p0 = best + scatter * rng.normal(size=(n_walkers, ndim))
    p0 = np.clip(p0, lo + 1e-9, hi - 1e-9)

    try:
        sampler = emcee.EnsembleSampler(n_walkers, ndim, log_prob)
        sampler.run_mcmc(p0, n_steps, progress=False)
        chain = sampler.get_chain(discard=n_burn, flat=True)
        return chain if len(chain) else None
    except Exception:
        return None

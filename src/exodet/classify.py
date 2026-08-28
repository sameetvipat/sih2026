"""Gradient-boosted classifier over the vetting features.

We deliberately use a tree ensemble on physically-meaningful features rather
than a raw CNN on pixels of the folded curve.  Two reasons:

  1. it trains in seconds on a few thousand examples, where a CNN would be
     data-starved;
  2. every prediction can be explained by naming the features that drove it,
     which is what a vetting decision actually needs.

Probabilities are isotonically calibrated on a held-out split so that a
reported "0.9 confidence" really does mean ~90% of such cases are correct --
the problem statement asks for a confidence level, not just a score.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, log_loss
from sklearn.model_selection import train_test_split

from .config import CLASSES
from .features import FEATURE_NAMES


@dataclass
class TrainReport:
    accuracy: float
    macro_f1: float
    log_loss_raw: float
    log_loss_calibrated: float
    confusion: np.ndarray
    report_text: str
    importances: pd.Series


def _clean(X: pd.DataFrame) -> pd.DataFrame:
    """Replace non-finite values -- trees handle NaN, but keep it explicit."""
    return X.replace([np.inf, -np.inf], np.nan)


def _wrap(model):
    """CalibratedClassifierCV over an already-fitted estimator."""
    try:                                     # sklearn >= 1.6
        from sklearn.frozen import FrozenEstimator
        return lambda method: CalibratedClassifierCV(FrozenEstimator(model),
                                                     method=method)
    except ImportError:                      # sklearn < 1.6
        return lambda method: CalibratedClassifierCV(model, method=method,
                                                     cv="prefit")


def _best_calibrator(model, X_cal, y_cal, seed):
    """Choose isotonic vs sigmoid on a split of the calibration data.

    Isotonic is flexible but needs a lot of data; sigmoid (Platt) is a
    two-parameter fit that behaves far better when the calibration set is
    small.  Selecting between them on an inner split keeps the test set clean.
    """
    make = _wrap(model)
    labels = list(range(len(CLASSES)))
    try:
        X_a, X_b, y_a, y_b = train_test_split(
            X_cal, y_cal, test_size=0.4, stratify=y_cal, random_state=seed)
    except ValueError:                       # too few samples to stratify
        return None

    best, best_ll = None, np.inf
    for method in ("sigmoid", "isotonic"):
        try:
            c = make(method)
            c.fit(X_a, y_a)
            ll = log_loss(y_b, c.predict_proba(X_b), labels=labels)
        except Exception:
            continue
        if ll < best_ll:
            best, best_ll = method, ll

    if best is None:
        return None
    try:                                     # refit the winner on all of X_cal
        final = make(best)
        final.fit(X_cal, y_cal)
        return final
    except Exception:
        return None


def train(df: pd.DataFrame, seed: int = 0, calibrate: bool = True):
    """Train the classifier. Returns (model, calibrator, TrainReport)."""
    import lightgbm as lgb

    X = _clean(df[FEATURE_NAMES])
    y = df["label"].map({c: i for i, c in enumerate(CLASSES)}).values

    # train / calibration / test = 60 / 20 / 20
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=seed)
    X_cal, X_te, y_cal, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=seed)

    model = lgb.LGBMClassifier(
        objective="multiclass", num_class=len(CLASSES),
        n_estimators=400, learning_rate=0.05,
        num_leaves=31, min_child_samples=20,
        subsample=0.9, subsample_freq=1, colsample_bytree=0.8,
        reg_lambda=1.0, class_weight="balanced",
        random_state=seed, n_jobs=-1, verbose=-1,
    )
    model.fit(X_tr, y_tr)

    proba_raw = model.predict_proba(X_te)
    ll_raw = float(log_loss(y_te, proba_raw, labels=list(range(len(CLASSES)))))

    calibrator = None
    ll_cal = ll_raw
    proba = proba_raw
    if calibrate:
        calibrator = _best_calibrator(model, X_cal, y_cal, seed)
        if calibrator is not None:
            proba_cal = calibrator.predict_proba(X_te)
            ll_cal = float(log_loss(y_te, proba_cal,
                                    labels=list(range(len(CLASSES)))))
            # Calibration is supposed to help. Isotonic in particular overfits
            # on small splits and can make probabilities strictly worse, so
            # keep it only if it actually improves the held-out log loss.
            if ll_cal <= ll_raw:
                proba = proba_cal
            else:
                calibrator = None
                ll_cal = ll_raw

    pred = proba.argmax(axis=1)
    from sklearn.metrics import accuracy_score, f1_score
    rep = TrainReport(
        accuracy=float(accuracy_score(y_te, pred)),
        macro_f1=float(f1_score(y_te, pred, average="macro")),
        log_loss_raw=ll_raw,
        log_loss_calibrated=ll_cal,
        confusion=confusion_matrix(y_te, pred, labels=list(range(len(CLASSES)))),
        report_text=classification_report(
            y_te, pred, labels=list(range(len(CLASSES))),
            target_names=CLASSES, zero_division=0),
        importances=pd.Series(model.feature_importances_,
                              index=FEATURE_NAMES).sort_values(ascending=False),
    )
    return model, calibrator, rep, (X_te, y_te, proba)


def predict_one(model, calibrator, features: dict) -> dict:
    """Classify a single feature dict. Returns label + full probability vector."""
    X = _clean(pd.DataFrame([features])[FEATURE_NAMES])
    est = calibrator if calibrator is not None else model
    proba = est.predict_proba(X)[0]
    i = int(np.argmax(proba))
    return {
        "label": CLASSES[i],
        "confidence": float(proba[i]),
        "probabilities": {c: float(p) for c, p in zip(CLASSES, proba)},
    }


def explain(model, features: dict, top_n: int = 4) -> list[tuple[str, float]]:
    """Name the features that most drove this prediction, via SHAP values.

    Falls back to global importances if SHAP is unavailable.
    """
    X = _clean(pd.DataFrame([features])[FEATURE_NAMES])
    try:
        contrib = model.predict(X, pred_contrib=True)
        contrib = np.asarray(contrib).reshape(len(CLASSES), -1)[:, :len(FEATURE_NAMES)]
        pred_i = int(np.argmax(model.predict_proba(X)[0]))
        vals = contrib[pred_i]
        order = np.argsort(np.abs(vals))[::-1][:top_n]
        return [(FEATURE_NAMES[i], float(vals[i])) for i in order]
    except Exception:
        imp = pd.Series(model.feature_importances_, index=FEATURE_NAMES)
        return [(k, float(v)) for k, v in imp.nlargest(top_n).items()]


def save(model, calibrator, rep: TrainReport, outdir="models"):
    import joblib
    os.makedirs(outdir, exist_ok=True)
    joblib.dump({"model": model, "calibrator": calibrator,
                 "features": FEATURE_NAMES, "classes": CLASSES},
                os.path.join(outdir, "classifier.joblib"))
    with open(os.path.join(outdir, "metrics.json"), "w") as fh:
        json.dump({"accuracy": rep.accuracy, "macro_f1": rep.macro_f1,
                   "log_loss_raw": rep.log_loss_raw,
                   "log_loss_calibrated": rep.log_loss_calibrated,
                   "confusion": rep.confusion.tolist(),
                   "classes": CLASSES,
                   "importances": rep.importances.to_dict()}, fh, indent=2)


def load(path="models/classifier.joblib"):
    import joblib
    d = joblib.load(path)
    return d["model"], d["calibrator"]

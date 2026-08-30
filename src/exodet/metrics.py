"""Standard reporting format for every evaluation in this project.

Raw accuracy alone is misleading here and increasingly so as real data volume
grows: catalogued false positives substantially outnumber confirmed planets, so
a model that simply predicts the majority class scores well while being useless
for the task.  Every real-data evaluation therefore reports per-class
precision, recall and F1 alongside macro F1, plus a confusion matrix.

Accuracy is also reported with a Wilson confidence interval.  On the real
held-out sets this project works with (tens of light curves, not thousands),
the interval is wide enough that differences between regimes are often not
distinguishable -- and a bare point estimate hides that completely.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)


def wilson_interval(correct: int, n: int, conf: float = 0.95):
    """Wilson score interval: better than the normal approximation at small n."""
    if n == 0:
        return (float("nan"), float("nan"))
    from scipy.stats import norm
    z = norm.ppf(1 - (1 - conf) / 2)
    p = correct / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return float(centre - half), float(centre + half)


def report(y_true, y_pred, classes, title: str = "", show_confusion: bool = True):
    """Print the standard block and return the headline numbers."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    acc = float(accuracy_score(y_true, y_pred))
    macro = float(f1_score(y_true, y_pred, average="macro"))
    lo, hi = wilson_interval(int(round(acc * n)), n)

    if title:
        print(f"\n--- {title} ---")
    print(f"accuracy {acc:.3f}  95% CI [{lo:.3f}, {hi:.3f}]   "
          f"macro F1 {macro:.3f}   (n={n})")
    print(classification_report(y_true, y_pred, labels=list(range(len(classes))),
                                target_names=classes, zero_division=0, digits=3))
    if show_confusion:
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
        print("confusion matrix (rows = true, cols = predicted)")
        print(f"{'':>10}" + "".join(f"{c[:8]:>9}" for c in classes))
        for name, row in zip(classes, cm):
            print(f"{name:>10}" + "".join(f"{v:>9d}" for v in row))
    return dict(accuracy=acc, macro_f1=macro, ci_low=lo, ci_high=hi, n=n)


def difference_ci(acc_a: float, acc_b: float, n: int, conf: float = 0.95):
    """CI on the difference between two accuracies measured on the same n.

    Returns (difference, low, high, distinguishable_from_zero).
    """
    from scipy.stats import norm
    z = norm.ppf(1 - (1 - conf) / 2)
    se = np.sqrt(acc_a * (1 - acc_a) / n + acc_b * (1 - acc_b) / n)
    d = acc_b - acc_a
    return float(d), float(d - z * se), float(d + z * se), bool(abs(d) > z * se)

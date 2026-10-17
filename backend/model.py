"""The shared fitted probability contract used for evaluation and serving."""
import numpy as np


def calibrated(bundle, x):
    raw = np.clip(bundle['model'].predict_proba(x[bundle['features']])[:, 1], 1e-6, 1-1e-6)
    return bundle['calibrator'].predict_proba(np.log(raw/(1-raw)).reshape(-1, 1))[:, 1]

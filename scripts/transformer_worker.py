"""One candidate model per process. Frames and predictions travel as files.

On this machine xgboost/lightgbm and torch cannot both initialize OpenMP in one
process (segfaults or hangs), so each worker imports only its own family and the
coordinator in transformer_sprint.py never imports a model library.
"""
import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

THREADS = 8
CHECKPOINT = Path.home() / 'Library/Caches/tabpfn' / 'tabpfn-v2-classifier-finetuned-zk73skhh.ckpt'


def build_model(name, train):
    if name == 'logistic_extended':
        return make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42))
    ratio = (len(train) - train.exceedance.sum()) / train.exceedance.sum()
    if name == 'xgboost_extended':
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=140, max_depth=3, learning_rate=.05, subsample=.8, colsample_bytree=.9,
                             scale_pos_weight=ratio, random_state=42, n_jobs=2)
    if name == 'lightgbm_extended':
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=140, max_depth=3, num_leaves=7, learning_rate=.05,
                              class_weight='balanced', verbosity=-1, random_state=42, n_jobs=2)
    if name == 'tabpfn_extended':
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion
        import torch
        torch.set_num_threads(THREADS)
        return TabPFNClassifier.create_default_for_version(ModelVersion.V2, device='cpu', random_state=42,
                                                    ignore_pretraining_limits=True)
    if name == 'ft_transformer_extended':
        from scripts.ft_transformer import FTTransformerClassifier
        return FTTransformerClassifier()
    raise ValueError('Unknown candidate: ' + name)


def calibrated_probability(model, calibrator, features, frame):
    raw = np.clip(model.predict_proba(frame[list(features)])[:, 1], 1e-6, 1 - 1e-6)
    return calibrator.predict_proba(np.log(raw / (1 - raw)).reshape(-1, 1))[:, 1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate')
    parser.add_argument('frames')
    parser.add_argument('out')
    parser.add_argument('--reuse', action='store_true', help='load the saved model and calibrator instead of fitting')
    parser.add_argument('--models', default=None, help='directory holding the frozen model/calibrator when reusing')
    args = parser.parse_args()
    frames_dir, out_dir = Path(args.frames), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    features = json.loads((frames_dir / 'features.json').read_text())['features']
    splits = json.loads((frames_dir / 'splits.json').read_text())['splits']
    models_dir = Path(args.models) if args.models else out_dir
    model_path, calibrator_path = models_dir / f'{args.candidate}.model.joblib', models_dir / f'{args.candidate}.calibrator.joblib'
    started = time.time()
    meta = dict(name=args.candidate, reuse=args.reuse, started_at=datetime.now(timezone.utc).isoformat())
    if args.reuse:
        if not model_path.exists() or not calibrator_path.exists():
            raise FileNotFoundError('Reuse requested but the frozen model is missing: ' + str(model_path))
        model, calibrator = joblib.load(model_path), joblib.load(calibrator_path)
        meta['fit_seconds'] = None
    else:
        train = pd.read_csv(frames_dir / 'train.csv')
        calibration = pd.read_csv(frames_dir / 'calibration.csv')
        model = build_model(args.candidate, train)
        if args.candidate == 'ft_transformer_extended':
            model.fit(train[features], train.exceedance, calibration[features], calibration.exceedance)
            meta['epochs_trained'] = int(model.epochs_trained_)
        else:
            model.fit(train[features], train.exceedance)
        raw = np.clip(model.predict_proba(calibration[features])[:, 1], 1e-6, 1 - 1e-6)
        calibrator = LogisticRegression(C=1, random_state=42).fit(np.log(raw / (1 - raw)).reshape(-1, 1), calibration.exceedance)
        joblib.dump(model, model_path)
        joblib.dump(calibrator, calibrator_path)
        meta['fit_seconds'] = round(time.time() - started, 1)
    for split in splits:
        frame = pd.read_csv(frames_dir / f'{split}.csv')
        probability = calibrated_probability(model, calibrator, features, frame)
        np.save(out_dir / f'{args.candidate}.{split}.npy', probability)
        meta.setdefault('splits', {})[split] = len(frame)
    meta['total_seconds'] = round(time.time() - started, 1)
    meta['checkpoint_sha256'] = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    (out_dir / f'{args.candidate}.meta.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(args.candidate, 'done in', meta['total_seconds'], 's', flush=True)


if __name__ == '__main__':
    main()


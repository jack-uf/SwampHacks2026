"""One multimodal candidate per process; frames and predictions travel as files.

Never imports two model families at once: xgboost/lightgbm and torch cannot share
one process on this machine. No calibration step in this pilot: probabilities are
reported as the model produced them.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

THREADS = 8
CHECKPOINT = Path.home() / 'Library/Caches/tabpfn' / 'tabpfn-v2-classifier-finetuned-zk73skhh.ckpt'


def build_model(name, train, features):
    ratio = (len(train) - train.exceedance.sum()) / train.exceedance.sum()
    if name.endswith('_tabpfn'):
        import torch
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion
        torch.set_num_threads(THREADS)
        return TabPFNClassifier.create_default_for_version(ModelVersion.V2, device='mps', random_state=42)
    if 'xgboost' in name:
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=140, max_depth=3, learning_rate=.05, subsample=.8, colsample_bytree=.9,
                             scale_pos_weight=ratio, random_state=42, n_jobs=2)
    if 'lightgbm' in name:
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=140, max_depth=3, num_leaves=7, learning_rate=.05,
                              class_weight='balanced', verbosity=-1, random_state=42, n_jobs=2)
    raise ValueError('Unknown candidate: ' + name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate')
    parser.add_argument('frames')
    parser.add_argument('out')
    args = parser.parse_args()
    frames_dir, out_dir = Path(args.frames), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    features = json.loads((frames_dir / 'features.json').read_text())['features']
    splits = json.loads((frames_dir / 'splits.json').read_text())['splits']
    train = pd.read_csv(frames_dir / 'train.csv')
    started = time.time()
    model = build_model(args.candidate, train, features)
    model.fit(train[features], train.exceedance)
    fit_seconds = round(time.time() - started, 1)
    meta = dict(name=args.candidate, features=len(features), rows=len(train), fit_seconds=fit_seconds)
    for split in splits:
        frame = pd.read_csv(frames_dir / f'{split}.csv')
        probability = model.predict_proba(frame[features])[:, 1]
        np.save(out_dir / f'{args.candidate}.{split}.npy', probability)
        meta[f'{split}_rows'] = len(frame)
    joblib.dump(dict(model=model, features=list(features), name=args.candidate), out_dir / f'{args.candidate}.model.joblib')
    meta['total_seconds'] = round(time.time() - started, 1)
    meta['checkpoint_sha256'] = hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() if CHECKPOINT.exists() else None
    (out_dir / f'{args.candidate}.meta.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(args.candidate, 'fit', fit_seconds, 's of', meta['total_seconds'], 's', flush=True)


if __name__ == '__main__':
    main()


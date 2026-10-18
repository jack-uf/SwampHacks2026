"""Neural worker for the subdaily signal cycles.

Torch is loaded in this process only. Loading it beside LightGBM in one process hangs or faults in
this environment, which is why every neural family in this repository runs as a subprocess.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from scripts.high_flow_signals_network import fit_network, network_logits

PARTITIONS = ['calibration', 'threshold', 'temporal_validation', 'geographic_validation']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job', help='JSON job describing features, input, output and bundle paths')
    arguments = parser.parse_args()
    job = json.loads(Path(arguments.job).read_text())
    table = pd.read_csv(job['input'], dtype={'station': str}, keep_default_na=False)
    features = job['features']
    train = table.loc[table.partition.eq('inner_train')].reset_index(drop=True)
    validation = table.loc[table.partition.eq('inner_validation')].reset_index(drop=True)
    bundle = fit_network(features, train, validation, seed=job['seed'], epochs=job['epochs'],
        patience=job['patience'])
    torch.save(bundle, job['bundle'])
    rows = []
    for key in PARTITIONS:
        part = table.loc[table.partition.eq(key)].reset_index(drop=True)
        if part.empty:
            raise ValueError('Missing partition: '+key)
        rows.append(part[['station', 'decision_time']].assign(partition=key,
            logit=network_logits(bundle, part)))
    pd.concat(rows, ignore_index=True).to_csv(job['output'], index=False,
        compression={'method': 'gzip', 'mtime': 0})
    print('worker', job['name'], 'epochs', bundle.get('epochs_run', 'n/a'),
        'log_loss', round(bundle['validation_log_loss'], 5), 'parameters', bundle['parameters'], flush=True)


if __name__ == '__main__':
    main()

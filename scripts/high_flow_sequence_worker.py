"""Sequence-model worker for the hourly high-flow experiment.

Torch is imported in this process only, following the repository's worker pattern: loading torch
beside LightGBM in one process hangs or faults in this environment, so every neural family here runs
as a subprocess. The job JSON names the stored per-gauge series, the row index of each partition, the
row positions of the inner training split and the architecture.

Windows are sliced from the stored series as they are needed and never written to disk: the machine
is disk-constrained, and the transient arrays live in this process's memory only. Channel statistics
come from inner-training windows only. Missing entries (absent or rejected weather) are filled at the
standardized origin and counted. Early stopping holds out a contiguous tail of the training period by
decision time, never calibration, threshold or 2024 rows.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from scripts import high_flow_sequence_windows as sequence


class CausalBlock(nn.Module):
    """Dilated causal convolution with a residual connection; step s sees only steps up to s."""

    def __init__(self, channels, kernel, dilation, dropout):
        super().__init__()
        self.left = (kernel - 1)*dilation
        self.conv = nn.Conv1d(channels, channels, kernel, dilation=dilation)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        padded = nn.functional.pad(x, (self.left, 0))
        return x + self.dropout(nn.functional.silu(self.conv(padded)[..., :x.shape[-1]]))


class SequenceModel(nn.Module):
    """Strided causal stem plus dilated causal blocks; the prediction reads the final step."""

    def __init__(self, channels, hidden, kernel, dilations, dropout, stride):
        super().__init__()
        self.left = kernel - 1
        self.stem = nn.Conv1d(channels, hidden, kernel, stride=stride)
        self.blocks = nn.ModuleList([CausalBlock(hidden, kernel, dilation, dropout)
                                     for dilation in dilations])
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        hidden = nn.functional.silu(self.stem(nn.functional.pad(x, (self.left, 0))))
        for block in self.blocks:
            hidden = block(hidden)
        return self.head(hidden[..., -1]).squeeze(1)


def partition_windows(directory, records, index_dir, key, cache):
    """The partition's index rows and its windows, sliced from the stored per-gauge series."""
    table = pd.read_csv(Path(index_dir)/(key + '.csv.gz'), dtype={'station': str, 'huc': str},
        keep_default_na=False)
    block = np.empty((len(table), len(sequence.CHANNELS), sequence.WINDOW), dtype=np.float32)
    times = pd.to_datetime(table.decision_time, utc=True)
    for station, group in table.groupby('station', sort=True):
        origin, series = sequence.load_series(directory, records, station, cache)
        block[group.index.to_numpy()] = sequence.windows(series, origin, times.loc[group.index])
    return table, block


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job', help='JSON job written by scripts/high_flow_sequence.py')
    arguments = parser.parse_args()
    job = json.loads(Path(arguments.job).read_text())
    settings, training = job['architecture'], job['training']
    records = json.loads(Path(job['series']).read_text())['series']
    torch.set_num_threads(job['threads'])
    torch.manual_seed(training['seed'])
    started = time.monotonic()
    cache = {}
    table, block = partition_windows(job['directory'], records, job['index'], job['partitions'][0], cache)
    inner = np.asarray(job['inner_train_rows'], dtype=int)
    tail = np.asarray(job['inner_validation_rows'], dtype=int)
    train, validation = block[inner], block[tail]
    del block
    targets = table.target.to_numpy('float32')
    y = torch.from_numpy(targets[inner])
    vy = torch.from_numpy(targets[tail])
    mean = train.mean(axis=(0, 2), keepdims=True)
    std = train.std(axis=(0, 2), keepdims=True)
    std = np.where(std < 1e-6, 1., std)

    def standardize(block):
        scaled = (block - mean)/std
        filled = int(np.isnan(scaled).sum())
        return torch.from_numpy(np.nan_to_num(scaled, nan=0., posinf=0., neginf=0.)), filled

    x, filled_train = standardize(train)
    vx, filled_validation = standardize(validation)
    del train, validation
    if len(torch.unique(y)) < 2 or len(torch.unique(vy)) < 2:
        raise ValueError('Require two-class inner training and validation rows')
    model = SequenceModel(len(job['channels']), settings['hidden'], settings['kernel'],
        settings['dilations'], settings['dropout'], settings['stem_stride'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=training['learning_rate'],
        weight_decay=training['weight_decay'])
    generator = torch.Generator().manual_seed(training['seed'])
    best_loss, best_state, stale, epochs_run = np.inf, None, 0, 0
    for epoch in range(training['epochs']):
        model.train()
        for batch in torch.randperm(len(y), generator=generator).split(training['batch']):
            optimizer.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(model(x[batch]), y[batch])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            value = float(nn.functional.binary_cross_entropy_with_logits(model(vx), vy))
        if value < best_loss - 1e-5:
            best_loss, stale = value, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= training['patience']:
                break
        epochs_run += 1
        print('epoch', epoch + 1, 'validation log loss', round(value, 5),
              'best', round(best_loss, 5), flush=True)
    if best_state is None:
        raise ValueError('Training produced no finite validation loss')
    model.load_state_dict(best_state)
    torch.save(dict(state=best_state, mean=mean, std=std, architecture=settings,
        channels=job['channels'], window=job['window'], validation_log_loss=best_loss,
        epochs_run=epochs_run), job['bundle'])
    model.eval()
    rows, filled = [], {}
    for key in job['scored']:
        scored, block = partition_windows(job['directory'], records, job['index'], key, cache)
        scaled, count = standardize(block)
        del block
        filled[key] = count
        outputs = []
        with torch.no_grad():
            for batch in torch.arange(len(scaled)).split(4096):
                outputs.append(model(scaled[batch]).numpy())
        rows.append(scored[['station', 'decision_time']].assign(partition=key,
            logit=np.concatenate(outputs)))
        del scored, scaled
    pd.concat(rows, ignore_index=True).to_csv(job['output'], index=False,
        compression={'method': 'gzip', 'mtime': 0})
    json.dump(dict(epochs_run=epochs_run, validation_log_loss=best_loss,
        parameters=sum(p.numel() for p in model.parameters()), seconds=time.monotonic() - started,
        inner_train_rows=len(y), inner_validation_rows=len(vy), filled_entries=dict(train=filled_train,
        validation=filled_validation, **filled)), Path(job['summary']).open('w'), indent=2)
    print('worker done: epochs', epochs_run, 'parameters', sum(p.numel() for p in model.parameters()),
          'validation log loss', round(best_loss, 5), 'seconds', round(time.monotonic() - started, 1),
          flush=True)


if __name__ == '__main__':
    main()

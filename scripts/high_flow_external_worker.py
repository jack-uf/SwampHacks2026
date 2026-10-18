"""Torch worker for the external Google-architecture high-flow comparison.

Torch is imported in this process only, following this repository's worker pattern: the driver
never loads torch, and the worker never loads LightGBM. The worker trains one candidate on the
inner training rows, early-stops on a contiguous tail of the training period, and writes logits
for the scored partitions.

The trunk adapts the Google release's encoder conventions (a per-timestep feature embedding, an
LSTM over the raw sequence, static embeddings, a linear head) to this repository's six-hour
decision population. The released Mean-Embedding-Forecast-LSTM itself cannot run on these caches:
it needs daily hres/graphcast/imerg/cpc arrays over 365-day sequences and 84 Multimet statics that
the caches do not contain (see experiments/high-flow-external-20261018/audit.json), so everything
here is labelled "Google architecture under our protocol", never Google's published performance.

Two candidates share the trunk:
  google_lstm_engineered  the raw hourly window plus the frozen cycle-11 engineered onset feature
                          set and the static attributes, each through its own embedding.
  google_lstm_raw         the raw hourly window only.
Engineered columns are rebuilt from the cached signals dataset through
scripts.high_flow_signals.population, the same single source of truth the frozen candidates used.
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
from scripts.high_flow_external import delay_discharge


class ExternalModel(nn.Module):
    """Per-step embedding, LSTM hindcast encoder, optional side embeddings, linear head."""

    def __init__(self, channels, architecture, engineered_dim, statics_dim):
        super().__init__()
        self.channel_embed = nn.Linear(channels, architecture['channel_hidden'])
        self.lstm = nn.LSTM(architecture['channel_hidden'], architecture['lstm_hidden'],
                            batch_first=True)
        parts = architecture['lstm_hidden']
        self.engineered = None
        if engineered_dim:
            self.engineered = nn.Linear(engineered_dim, architecture['engineered_hidden'])
            parts += architecture['engineered_hidden']
        self.statics = None
        if statics_dim:
            self.statics = nn.Linear(statics_dim, architecture['statics_hidden'])
            parts += architecture['statics_hidden']
        self.head = nn.Sequential(nn.Linear(parts, architecture['head_hidden']), nn.SiLU(),
            nn.Dropout(architecture['dropout']), nn.Linear(architecture['head_hidden'], 1))

    def forward(self, window, engineered, statics):
        embedded = torch.tanh(self.channel_embed(window.transpose(1, 2)))
        output, _ = self.lstm(embedded)
        parts = [output[:, -1]]
        if self.engineered is not None:
            parts.append(torch.tanh(self.engineered(engineered)))
        if self.statics is not None:
            parts.append(torch.tanh(self.statics(statics)))
        return self.head(torch.cat(parts, dim=-1)).squeeze(1)


def index_table(index_dir, key):
    return pd.read_csv(Path(index_dir)/(key + '.csv.gz'), dtype={'station': str, 'huc': str},
                       keep_default_na=False)


def partition_windows(directory, records, table, cache):
    """The partition's windows, sliced from the stored per-gauge series.

    The discharge channel is delayed two hours through the driver helper so the newest value in
    every window equals the frozen candidate's q_2; idem for the weather channels, which keep the
    six-hour receipt delay the sequence sprint already applied.
    """
    block = np.empty((len(table), len(sequence.CHANNELS), sequence.WINDOW), dtype=np.float32)
    times = pd.to_datetime(table.decision_time, utc=True)
    for station, group in table.groupby('station', sort=True):
        origin, series = sequence.load_series(directory, records, station, cache)
        block[group.index.to_numpy()] = sequence.windows(delay_discharge(series), origin,
            times.loc[group.index])
    return block


def side_matrix(frame, table, columns):
    """Engineered and static columns for the partition's rows, joined by row identity."""
    if not columns:
        return None
    # Labels repeat on purpose (the statics are also part of the engineered set and must reach the
    # network a second time through their own embedding), and pandas list selection multiplies
    # repeated labels, so select by position instead.
    unique = list(dict.fromkeys(columns))
    merged = table[['station', 'decision_time']].merge(
        frame[['station', 'decision_time'] + unique],
        on=['station', 'decision_time'], how='left', validate='one_to_one')
    if len(merged) != len(table):
        raise ValueError('Side feature join changed the row count')
    values = merged[unique].to_numpy(np.float32)
    return values[:, [unique.index(name) for name in columns]]


def window_stats(block):
    values = block
    mean = np.nanmean(values, axis=(0, 2), keepdims=True)
    std = np.nanstd(values, axis=(0, 2), keepdims=True)
    mean = np.where(np.isfinite(mean), mean, 0.)
    std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_window_stats(block, mean, std):
    scaled = (block - mean)/std
    counts = [int(np.isnan(scaled[:, i]).sum()) for i in range(scaled.shape[1])]
    return np.nan_to_num(scaled, nan=0., posinf=0., neginf=0.).astype(np.float32), counts


def matrix_stats(matrix):
    mean = np.nanmean(matrix, axis=0)
    std = np.nanstd(matrix, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.)
    std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_matrix_stats(matrix, mean, std, names):
    scaled = (matrix - mean)/std
    missing = np.isnan(scaled)
    filled = {name: int(missing[:, i].sum()) for i, name in enumerate(names)
              if missing[:, i].any()}
    return np.nan_to_num(scaled, nan=0., posinf=0., neginf=0.).astype(np.float32), filled


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job', help='JSON job written by scripts/high_flow_external.py')
    arguments = parser.parse_args()
    job = json.loads(Path(arguments.job).read_text())
    settings, training = job['architecture'], job['training']
    records = json.loads(Path(job['series']).read_text())['series']
    torch.set_num_threads(job['threads'])
    torch.manual_seed(training['seed'])
    started = time.monotonic()
    cache = {}
    engineered_columns = list(job['engineered_features'])
    statics_columns = list(job['statics'])
    side_columns = engineered_columns + statics_columns
    from scripts import high_flow_signals as signals
    frame = signals.population(job['engineered_cycle']) if side_columns else None
    table = index_table(job['index'], job['partitions'][0])
    block = partition_windows(job['directory'], records, table, cache)
    inner = np.asarray(job['inner_train_rows'], dtype=int)
    tail = np.asarray(job['inner_validation_rows'], dtype=int)
    train, validation = block[inner], block[tail]
    del block
    targets = table.target.to_numpy('float32')
    y = torch.from_numpy(targets[inner])
    vy = torch.from_numpy(targets[tail])
    window_mean, window_std = window_stats(train)
    x, train_fill = apply_window_stats(train, window_mean, window_std)
    vx, validation_fill = apply_window_stats(validation, window_mean, window_std)
    del train, validation
    x, vx = torch.from_numpy(x), torch.from_numpy(vx)
    side = side_matrix(frame, table, side_columns)
    engineered_dim, statics_dim = len(engineered_columns), len(statics_columns)
    fills = dict(windows=dict(train=train_fill, validation=validation_fill, scored={}),
                 engineered=dict(), statics=dict())
    if engineered_dim:
        engineered_train = side[inner, :engineered_dim]
        engineered_mean, engineered_std = matrix_stats(engineered_train)
        engineered_matrix, filled = apply_matrix_stats(side[:, :engineered_dim], engineered_mean,
            engineered_std, engineered_columns)
        fills['engineered']['train'] = filled
        engineered_inner = torch.from_numpy(engineered_matrix[inner])
        engineered_validation = torch.from_numpy(engineered_matrix[tail])
    else:
        engineered_matrix, engineered_inner, engineered_validation = None, None, None
    if statics_dim:
        statics_train = side[inner, engineered_dim:]
        statics_mean, statics_std = matrix_stats(statics_train)
        statics_matrix, filled = apply_matrix_stats(side[:, engineered_dim:], statics_mean,
            statics_std, statics_columns)
        fills['statics']['train'] = filled
        statics_inner = torch.from_numpy(statics_matrix[inner])
        statics_validation = torch.from_numpy(statics_matrix[tail])
    else:
        statics_matrix, statics_inner, statics_validation = None, None, None
    if len(torch.unique(y)) < 2 or len(torch.unique(vy)) < 2:
        raise ValueError('Require two-class inner training and validation rows')
    model = ExternalModel(len(job['channels']), settings, engineered_dim, statics_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training['learning_rate'],
        weight_decay=training['weight_decay'])
    generator = torch.Generator().manual_seed(training['seed'])
    best_loss, best_state, stale, epochs_run = np.inf, None, 0, 0
    for epoch in range(training['epochs']):
        model.train()
        for batch in torch.randperm(len(y), generator=generator).split(training['batch']):
            optimizer.zero_grad()
            logits = model(x[batch], engineered_inner[batch] if engineered_dim else None,
                statics_inner[batch] if statics_dim else None)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, y[batch])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            value = float(nn.functional.binary_cross_entropy_with_logits(
                model(vx, engineered_validation if engineered_dim else None,
                    statics_validation if statics_dim else None), vy))
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
    torch.save(dict(state=best_state, window_mean=window_mean, window_std=window_std,
        architecture=settings, channels=job['channels'], window=job['window'],
        engineered_features=engineered_columns, statics=statics_columns,
        validation_log_loss=best_loss, epochs_run=epochs_run, fills=fills), job['bundle'])
    model.eval()
    rows = []
    for key in job['scored']:
        scored_table = index_table(job['index'], key)
        scored_block = partition_windows(job['directory'], records, scored_table, cache)
        scaled, fills['windows']['scored'][key] = apply_window_stats(scored_block, window_mean,
            window_std)
        del scored_block
        if engineered_dim:
            engineered_scored, filled = apply_matrix_stats(
                side_matrix(frame, scored_table, engineered_columns), engineered_mean,
                engineered_std, engineered_columns)
            fills['engineered'][key] = filled
        else:
            engineered_scored = None
        if statics_dim:
            statics_scored, filled = apply_matrix_stats(
                side_matrix(frame, scored_table, statics_columns), statics_mean, statics_std,
                statics_columns)
            fills['statics'][key] = filled
        else:
            statics_scored = None
        tensor = torch.from_numpy(scaled)
        engineered_tensor = torch.from_numpy(engineered_scored) if engineered_dim else None
        statics_tensor = torch.from_numpy(statics_scored) if statics_dim else None
        outputs = []
        with torch.no_grad():
            for batch in torch.arange(len(tensor)).split(4096):
                outputs.append(model(tensor[batch],
                    engineered_tensor[batch] if engineered_dim else None,
                    statics_tensor[batch] if statics_dim else None).numpy())
        rows.append(scored_table[['station', 'decision_time']].assign(partition=key,
            logit=np.concatenate(outputs)))
        del scaled, tensor, engineered_tensor, statics_tensor
    pd.concat(rows, ignore_index=True).to_csv(job['output'], index=False,
        compression={'method': 'gzip', 'mtime': 0})
    summary = dict(epochs_run=epochs_run, validation_log_loss=best_loss,
        parameters=sum(p.numel() for p in model.parameters()), seconds=time.monotonic() - started,
        inner_train_rows=len(y), inner_validation_rows=len(vy), fills=fills)
    json.dump(summary, Path(job['summary']).open('w'), indent=2)
    print('worker done:', job['name'], 'epochs', epochs_run, 'parameters', summary['parameters'],
          'validation log loss', round(best_loss, 5), 'seconds', round(summary['seconds'], 1),
          flush=True)


if __name__ == '__main__':
    main()

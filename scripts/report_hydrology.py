"""Audit frozen outputs and render research tables without fitting any model."""
import json
import os

import numpy as np
import pandas as pd

from scripts.hydrology_research import OUT, ROOT, digest, partitions, protocol, write_json
from scripts.hydrology_worker import metrics


def same_numbers(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys(), 'Metric fields changed'
        for key in expected:
            same_numbers(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected), 'Metric bins changed'
        for a, e in zip(actual, expected):
            same_numbers(a, e)
    elif isinstance(expected, (int, float)):
        assert np.isclose(actual, expected, rtol=1e-10, atol=1e-12), (actual, expected)
    else:
        assert actual == expected, (actual, expected)


def main():
    protocol()
    selection = json.loads((OUT / 'selection.json').read_text())
    provenance = json.loads((OUT / 'provenance.json').read_text())
    assert digest(OUT / 'training.csv.gz') == selection['dataset_sha256'] == provenance['dataset_sha256']
    for path, checksum in selection['code'].items():
        assert digest(ROOT / path) == checksum, path
    candidates = {c['name']: c for c in selection['candidates']}
    data = pd.read_csv(OUT / 'training.csv.gz', dtype={'station': str, 'huc': str})
    parts = {h: partitions(data[data.horizon.eq(h)]) for h in (1, 2)}
    summary, predictions = [], {}
    for family in ['baseline', 'logistic', 'xgboost', 'lightgbm']:
        for result in json.loads((OUT / f'{family}-evaluation.json').read_text())['candidates']:
            name = result['name']; candidate = candidates[name]
            assert digest(OUT / 'artifacts' / f'{name}.joblib') == candidate['artifact_sha256']
            for split, measured in result['tests'].items():
                path = OUT / 'predictions' / f'{name}.{split}.csv.gz'
                frame = pd.read_csv(path, dtype={'station': str, 'huc': str}, float_precision='round_trip')
                expected = parts[candidate['horizon']][split]
                keys = ['station', 'huc', 'target_date', 'exceedance', 'onset']
                pd.testing.assert_frame_equal(frame[keys].reset_index(drop=True), expected[keys].reset_index(drop=True))
                checked = metrics(frame, frame.probability.to_numpy(), candidate['thresholds'])
                same_numbers(checked, {key: measured[key] for key in checked})
                predictions[str(path.relative_to(OUT))] = digest(path)
                row = dict(candidate=name, split=split, horizon=candidate['horizon'],
                    selected=name == selection['selected'][str(candidate['horizon'])],
                    **{key: measured[key] for key in ['rows', 'positives', 'ap', 'brier', 'log_loss', 'macro_station_ap']})
                for budget in ['1', '5']:
                    row.update({f'{key}_budget{budget}': value for key, value in measured['alerts'][budget].items()})
                summary.append(row)
    pd.DataFrame(summary).to_csv(OUT / 'summary.csv', index=False)
    data.assign(year=data.target_date.str[:4]).groupby(['station', 'role', 'huc', 'horizon', 'year']).agg(
        rows=('exceedance', 'size'), positives=('exceedance', 'sum'), onsets=('onset', 'sum')).to_csv(OUT / 'coverage.csv')
    write_json(OUT / 'audit.json', dict(selection_sha256=digest(OUT / 'selection.json'),
        protocol_sha256=digest(OUT / 'protocol.json'), dataset_sha256=digest(OUT / 'training.csv.gz'),
        summary_sha256=digest(OUT / 'summary.csv'), predictions=predictions,
        verification='All 56 aggregate evaluations recomputed from saved probabilities; cohort identities checked against frozen dataset'))
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/planit-hydrology-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    table = pd.DataFrame(summary)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), layout='constrained')
    labels = ['Persistence', 'Weather + hydrology', '+ rainfall memory', '+ snow-aware memory']
    for j, horizon in enumerate([1, 2]):
        names = [f'baseline_persistence_h{horizon}', f'lightgbm_combined_h{horizon}',
                 f'lightgbm_memory_h{horizon}', f'lightgbm_snow_h{horizon}']
        for i, split in enumerate(['forward_test', 'geographic_test']):
            rows = table[table.split.eq(split)].set_index('candidate').loc[names]
            ax = axes[i, j]
            bars = ax.barh(labels[::-1], rows.ap.to_numpy()[::-1], color=['#174c48', '#82aaa0', '#b9cbc3', '#d9dfd8'])
            ax.bar_label(bars, fmt='%.3f', padding=4)
            ax.set(xlim=(0, 1), xlabel='Average precision', title=f'{"Forward time" if i == 0 else "Unseen western watersheds"}, D+{horizon}')
            ax.grid(axis='x', alpha=.15); ax.set_axisbelow(True)
    fig.suptitle('High-discharge forecasting: frozen final tests', fontsize=15)
    fig.savefig(OUT / 'comparison.png', dpi=160)
    plt.close(fig)
    print('Audited 28 candidates, 56 test cohorts; tables and figure saved')


if __name__ == '__main__':
    main()

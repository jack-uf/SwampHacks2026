"""Reconstruct final onset labels and audit frozen historical-transition results."""
import json

import numpy as np
import pandas as pd
from shapely.geometry import shape

from scripts.analog_research import ROOT, OUT, ANALOG, analog_features, load_data, digest, write_json
from scripts.analog_worker import verify_selection
from scripts.hydrology_research import daily_discharge
from scripts.onset_worker import metrics


def audit():
    selection = verify_selection()
    sources = {}
    for split in ['training', 'test']:
        provenance = json.loads((OUT / f'{split}.provenance.json').read_text())
        assert provenance['protocol_sha256'] == digest(OUT / 'protocol.json')
        for path, record in provenance['sources'].items():
            source = ROOT / path
            assert source.stat().st_size == record['bytes'] and record['url'].startswith('https://')
            assert digest(source) == record['sha256'], path
            assert json.loads(source.with_name(source.name + '.request.json').read_text()) == record
            sources[path] = record
    train, test = load_data('training'), load_data('test')
    original = pd.read_csv(ROOT / 'experiments/sprint-20261017-response/training.csv.gz',
                           dtype={'station': str, 'huc': str}).fillna({'event_date': ''})
    keys = ['station', 'decision_date']
    pd.testing.assert_frame_equal(train[original.columns].sort_values(keys).reset_index(drop=True),
                                  original.sort_values(keys).reset_index(drop=True))
    registry = {s['station']: s for s in json.loads((OUT / 'stations.json').read_text())}
    coverage = json.loads((OUT / 'test.provenance.json').read_text())['coverage']
    assert len(registry) == len(coverage) == 31
    assert set(test.station) == {r['station'] for r in coverage if 'excluded' not in r}
    assert not test.station.isin(train.station).any() and test.role.eq('fresh_test').all()
    assert test.decision_date.between('2023-01-08', '2024-09-25').all()
    assert test.huc.str[:4].nunique() == test.station.nunique()
    for sid, frame in test.groupby('station'):
        path = next(ROOT / p for p in sources if p.endswith(f'/sources/{sid}.rdb'))
        q = daily_discharge(path.read_text(), sid)
        q.index = pd.to_datetime(q.index)
        threshold = q.loc[q.index < '2016-01-01'].quantile(.95)
        issue = pd.DatetimeIndex(frame.decision_date)
        assert np.allclose(frame.reference_cfs, threshold)
        for lag in [3, 4, 7, 14]:
            assert np.allclose(frame[f'q_{lag}'], np.log1p(q.reindex(issue - pd.Timedelta(days=lag)).to_numpy() / threshold))
        # Enumerate calendar days independently of the rolling-window label builder.
        history = np.column_stack([q.reindex(issue + pd.Timedelta(days=d)).to_numpy() for d in range(-9, 4)])
        assert np.isfinite(history).all() and (history[:, :7] <= threshold).all()
        events = np.full(len(frame), '', dtype=object)
        for lead in [3, 2, 1]:
            high = history[:, 9 + lead] > threshold
            quiet = (history[:, 9 + lead - 7:9 + lead] <= threshold).all(axis=1)
            mask = high & quiet
            events[mask] = (issue[mask] + pd.Timedelta(days=lead)).strftime('%Y-%m-%d')
            assert np.allclose(frame[f'flow_{lead}'], np.log1p(history[:, 9 + lead] / threshold))
        assert np.array_equal(events, frame.event_date) and np.array_equal(events != '', frame.exceedance.astype(bool))
        positive = events != ''
        local = pd.DatetimeIndex(events[positive]).tz_localize(registry[sid]['timezone']).tz_convert('UTC').tz_localize(None)
        assert np.allclose((local - issue[positive]).total_seconds() / 3600, frame.loc[positive, 'event_lead_hours'])
        sample = frame.iloc[sorted({0, len(frame)//2, len(frame)-1})]
        expected, metadata = analog_features(q, pd.DatetimeIndex(sample.decision_date))
        assert np.allclose(expected, sample[ANALOG])
        assert metadata == next(r['reference'] for r in coverage if r['station'] == sid)
        assert metadata['last_library_target'] < '2016-01-01'
    upstream = ROOT / 'experiments/sprint-20261017-upstream'
    indexed = json.loads((upstream / 'audit.json').read_text())['sources']
    geography = json.loads((ROOT / 'experiments/sprint-20261017-response/geography.json').read_text())
    occupied = []
    for sid, checksum in geography['boundary_hashes'].items():
        path = ROOT / f'experiments/sprint-20261017-response/basins/{sid}.gauge-basin.json'
        assert digest(path) == checksum
        occupied.append(shape(json.loads(path.read_text())['features'][0]['geometry']))
    for sid in registry:
        path = upstream / 'basins' / f'{sid}.json'
        assert digest(path) == indexed[str(path.relative_to(ROOT))]['sha256']
        polygon = shape(json.loads(path.read_text())['features'][0]['geometry'])
        assert polygon.is_valid and polygon.area > 0
        assert all(polygon.intersection(other).area == 0 for other in occupied)
        occupied.append(polygon)
    report = json.loads((OUT / 'evaluation.json').read_text())
    assert report['selection_sha256'] == digest(OUT / 'selection.json')
    for family, checksum in report['evaluation_hashes'].items():
        path = OUT / f'{family}.evaluation.json'
        assert digest(path) == checksum
        evaluation = json.loads(path.read_text())
        assert evaluation['selection_sha256'] == digest(OUT / 'selection.json')
        assert evaluation['test_provenance_sha256'] == digest(OUT / 'test.provenance.json')
    predictions = {}
    for row in report['candidates']:
        name = row['name']; path = OUT / 'predictions' / f'{name}.test.csv.gz'
        f = pd.read_csv(path, dtype={'station': str, 'huc': str}, float_precision='round_trip').fillna({'event_date': ''})
        pd.testing.assert_frame_equal(f[test.columns.intersection(f.columns)], test[test.columns.intersection(f.columns)])
        record = next(r for r in selection['candidates'] if r['name'] == name)
        for policy, threshold in record['thresholds'].items():
            assert np.array_equal(f[policy], f.probability >= threshold)
        assert not f.strict_persistence.any()
        alerts = {k: f[k].to_numpy() for k in row['overall']['policies']}
        actual = metrics(test, f.probability.to_numpy(), alerts)
        # Preserve probability ties when parsing decimal predictions for scoring.
        for key in ['ap', 'brier', 'log_loss']:
            assert np.isclose(actual.pop(key), row['overall'][key], rtol=1e-12, atol=1e-14)
        actual.pop('reliability')
        assert actual == {k: v for k, v in row['overall'].items() if k not in ['ap', 'brier', 'log_loss', 'reliability']}
        predictions[name] = digest(path)
    write_json(OUT / 'audit.json', dict(status='passed', code_sha256=digest(ROOT / 'scripts/audit_analog.py'),
        source_files=len(sources), source_bytes=sum(r['bytes'] for r in sources.values()),
        new_source_bytes=sum(r['bytes'] for p, r in sources.items() if p.startswith(str(OUT.relative_to(ROOT)))),
        development_rows=len(train), test_rows=len(test), test_gauges=int(test.station.nunique()),
        reserved_basins=len(registry), reference_boundary_files=len(geography['boundary_hashes']),
        reconstructed_labels=len(test), analog_queries_checked=3 * int(test.station.nunique()),
        predictions=predictions, evaluation_sha256=digest(OUT / 'evaluation.json')))
    print('Audited sources, independent labels, basin separation, predictions and frozen inputs', flush=True)


if __name__ == '__main__':
    audit()

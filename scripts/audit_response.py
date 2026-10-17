"""Read-only source-label and frozen-artifact checks, one model family per process."""
import argparse
import json
import platform

import joblib
import numpy as np
import pandas as pd
from shapely.geometry import shape
from sklearn.metrics import average_precision_score, brier_score_loss

from scripts.hydrology_research import daily_discharge
from scripts.response_research import ROOT, OUT, digest, write_json, load_data, flow_signature
from scripts.response_worker import CLASSIC, NEURAL, predict, verify_selection


def audit(family):
    selection = verify_selection()
    provenance = json.loads((OUT / 'provenance.json').read_text())
    evaluation = json.loads((OUT / 'evaluation.json').read_text())
    assert evaluation['selection_sha256'] == digest(OUT / 'selection.json')
    for name, checksum in selection['family_hashes'].items():
        assert digest(OUT / f'{name}.selection.json') == checksum
    test = load_data('test').reset_index(drop=True)
    registry = {s['station']: s for s in json.loads((OUT / 'stations.json').read_text())}
    checked = 0
    if family == 'classic':
        geography = json.loads((OUT / 'geography.json').read_text())
        polygons = {}
        for sid, checksum in geography['boundary_hashes'].items():
            path = OUT / 'basins' / f'{sid}.upstream-4dp.json'
            assert digest(path) == checksum
            polygons[sid] = shape(json.loads(path.read_text())['features'][0]['geometry'])
            assert polygons[sid].is_valid and polygons[sid].area > 0
            assert digest(OUT / 'basins' / f'{sid}.location-4dp.json') == geography['location_hashes'][sid]
        for sid in test.station.unique():
            for other, polygon in polygons.items():
                if other in registry and registry[other]['role'] == 'fresh_test' and registry[other]['huc'][:4] == registry[sid]['huc'][:4]:
                    continue
                assert polygons[sid].intersection(polygon).area == 0, (sid, other)
        for sid, group in test.groupby('station'):
            source = next(ROOT / p for p in provenance['sources'] if p.endswith('/' + sid + '.rdb'))
            assert digest(source) == provenance['sources'][str(source.relative_to(ROOT))]['sha256']
            q = daily_discharge(source.read_text(), sid); q.index = pd.to_datetime(q.index)
            signature = flow_signature(q, registry[sid]['area_km2'])
            assert signature == provenance['signatures'][sid]
            reference = q[q.index < '2016-01-01'].quantile(.95)
            for row in group.itertuples():
                issue = pd.Timestamp(row.decision_date)
                quiet = q.reindex(pd.date_range(issue - pd.Timedelta(days=9), periods=7))
                assert quiet.notna().all() and quiet.le(reference).all()
                assert q.reindex(pd.date_range(issue - pd.Timedelta(days=6), periods=10)).notna().all()
                event = ''
                for lead in [1, 2, 3]:
                    day = issue + pd.Timedelta(days=lead)
                    before = q.reindex(pd.date_range(day - pd.Timedelta(days=7), periods=7))
                    if q.loc[day] > reference and before.notna().all() and before.le(reference).all():
                        event = str(day.date()); break
                assert row.event_date == event and row.exceedance == bool(event)
                if event:
                    start = pd.Timestamp(event).tz_localize(registry[sid]['timezone']).tz_convert('UTC')
                    assert row.event_lead_hours == (start - issue.tz_localize('UTC')).total_seconds() / 3600
                for lag in [3, 4, 7, 14]:
                    assert np.isclose(getattr(row, f'q_{lag}'), np.log1p(q.loc[issue - pd.Timedelta(days=lag)] / reference))
                checked += 1
    records = {r['name']: r for r in evaluation['candidates']}
    reproduced = []
    for candidate in selection['candidates']:
        name = candidate['name']
        if name not in (CLASSIC if family == 'classic' else NEURAL):
            continue
        artifact = OUT / 'artifacts' / f'{name}.joblib'
        assert digest(artifact) == candidate['artifact_sha256']
        saved = pd.read_csv(OUT / 'predictions' / f'{name}.test.csv.gz', dtype={'station': str, 'huc': str}).fillna({'event_date': ''})
        assert saved[['station', 'decision_date', 'event_date', 'exceedance']].equals(test[['station', 'decision_date', 'event_date', 'exceedance']])
        p = saved.probability.to_numpy(); y = saved.exceedance.to_numpy(bool)
        assert np.allclose(predict(joblib.load(artifact), test), p, rtol=0, atol=1e-14)
        overall = records[name]['overall']
        assert np.isclose(average_precision_score(y, p), overall['ap'])
        assert np.isclose(brier_score_loss(y, p), overall['brier'])
        for policy, result in overall['policies'].items():
            alerts = saved[policy].to_numpy(bool)
            if policy in candidate['thresholds']:
                assert np.array_equal(alerts, p >= candidate['thresholds'][policy])
            events = saved[y][['station', 'event_date']].drop_duplicates()
            hits = saved[y & alerts][['station', 'event_date']].drop_duplicates()
            assert len(events) == result['events']
            assert np.isclose(len(hits) / len(events), result['event_recall'])
            assert int(np.sum(alerts & ~y)) == result['false_alerts']
        reproduced.append(name)
    write_json(OUT / f'{family}.audit.json', dict(status='passed', label_rows_reconstructed=checked,
        candidates_reproduced=reproduced, evaluation_sha256=digest(OUT / 'evaluation.json'),
        audit_code_sha256=digest(ROOT / 'scripts/audit_response.py'), platform=platform.platform(),
        machine=platform.machine(), python=platform.python_version(), production_preserved=True))
    print('Audited', family, checked, 'source labels and', len(reproduced), 'candidate predictions')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('family', choices=['classic', 'neural'])
    audit(parser.parse_args().family)

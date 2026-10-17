"""Independent source-label reconstruction and frozen warning replay checks."""
import argparse
import json
import platform

import joblib
import numpy as np
import pandas as pd
from shapely.geometry import shape
from sklearn.metrics import average_precision_score, brier_score_loss

from scripts.hydrology_research import daily_discharge
from scripts.response_research import OUT as PREVIOUS, flow_signature
from scripts.warning_research import ROOT, OUT, digest, write_json, load_test, budget_alerts
from scripts.warning_worker import FAMILIES, predict, policies, verify_selection


def audit(family):
    selection = verify_selection()
    evaluation = json.loads((OUT / 'evaluation.json').read_text())
    assert evaluation['selection_sha256'] == digest(OUT / 'selection.json')
    test = load_test().reset_index(drop=True)
    provenance = json.loads((OUT / 'provenance.json').read_text())
    registry = {s['station']: s for s in json.loads((OUT / 'stations.json').read_text())}
    checked = 0
    if family == 'classic':
        for path, record in provenance['sources'].items():
            assert digest(ROOT / path) == record['sha256'], path
        geography = json.loads((PREVIOUS / 'geography.json').read_text())
        polygons = {}
        for sid, checksum in geography['boundary_hashes'].items():
            path = PREVIOUS / 'basins' / f'{sid}.gauge-basin.json'
            assert digest(path) == checksum
            polygons[sid] = shape(json.loads(path.read_text())['features'][0]['geometry'])
            assert polygons[sid].is_valid and polygons[sid].area > 0
            assert digest(PREVIOUS / 'basins' / f'{sid}.gauge-location.json') == geography['location_hashes'][sid]
        previous = {s['station']: s for name in ['hydrology', 'onset', 'response']
                    for s in json.loads((ROOT / f'experiments/sprint-20261017-{name}/stations.json').read_text())}
        assert not set(previous) & set(registry)
        assert not {s['huc'][:4] for s in previous.values()} & {s['huc'][:4] for s in registry.values()}
        for sid in registry:
            for other in set(previous) | set(registry):
                if other in registry and registry[other]['huc'][:4] == registry[sid]['huc'][:4]:
                    continue
                assert polygons[sid].intersection(polygons[other]).area == 0, (sid, other)
        for sid, group in test.groupby('station'):
            source = next(ROOT / p for p in provenance['sources'] if p.endswith('/' + sid + '.rdb'))
            q = daily_discharge(source.read_text(), sid); q.index = pd.to_datetime(q.index)
            reference = q[q.index < '2016-01-01'].quantile(.95)
            signature = flow_signature(q, registry[sid]['area_km2'])
            for key, value in signature.items():
                assert np.allclose(group[key], value, rtol=1e-14)
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
    reproduced = []
    records = {r['name']: r for r in evaluation['candidates']}
    for r in selection['candidates']:
        name = r['name']
        if name not in FAMILIES[family]:
            continue
        artifact = ROOT / r['artifact']
        assert digest(artifact) == r['artifact_sha256']
        bundle = joblib.load(artifact)
        saved = pd.read_csv(OUT / 'predictions' / f'{name}.test.csv.gz', dtype={'station': str, 'huc': str}).fillna({'event_date': ''})
        assert saved[['station', 'decision_date', 'event_date', 'exceedance']].equals(test[['station', 'decision_date', 'event_date', 'exceedance']])
        p = predict(bundle, test)
        assert np.allclose(p, saved.probability, atol=1e-14, rtol=0)
        report = records[name]['overall']
        assert np.isclose(average_precision_score(test.exceedance, p), report['ap'])
        assert np.isclose(brier_score_loss(test.exceedance, p), report['brier'])
        replay = policies(test, p, r['thresholds'])
        for policy, mask in replay.items():
            assert np.array_equal(mask, saved[policy])
            result = report['policies'][policy]
            assert int((mask & test.exceedance.eq(0)).sum()) == result['false_alerts']
            hits = test[mask & test.exceedance.eq(1)][['station', 'event_date']].drop_duplicates()
            events = test[test.exceedance.eq(1)][['station', 'event_date']].drop_duplicates()
            assert len(events) == result['events']
            assert np.isclose(len(hits) / len(events), result['event_recall'])
            if policy.startswith('bucket'):
                counts = test[['huc', 'decision_date']].assign(false=mask & test.exceedance.eq(0), rows=1)
                for _, region in counts.groupby(counts.huc.str[:2]):
                    daily = region.groupby('decision_date')[['false', 'rows']].sum().cumsum()
                    assert (daily.false * 100 <= daily.rows).all()
                changed = test.copy()
                recent = pd.to_datetime(changed.decision_date) > pd.to_datetime(changed.decision_date).max() - pd.Timedelta(days=10)
                changed.loc[recent, 'exceedance'] = 1 - changed.loc[recent, 'exceedance']
                assert np.array_equal(mask, budget_alerts(changed, p, r['thresholds']['static_' + policy[-1]]))
        if name == 'day':
            old = joblib.load(PREVIOUS / 'artifacts/pooled.joblib')
            assert all(np.array_equal(value.numpy(), bundle['members'][0]['network'][key].numpy()) for key, value in old['network'].items())
        reproduced.append(name)
    write_json(OUT / f'{family}.audit.json', dict(status='passed', source_labels_reconstructed=checked,
        candidates_reproduced=reproduced, evaluation_sha256=digest(OUT / 'evaluation.json'),
        audit_code_sha256=digest(ROOT / 'scripts/audit_warning.py'), platform=platform.platform()))
    print('Audited', family, checked, 'source labels and', len(reproduced), 'candidate predictions', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('family', choices=list(FAMILIES))
    audit(parser.parse_args().family)

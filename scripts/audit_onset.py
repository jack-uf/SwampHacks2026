"""Read-only reconstruction of fresh event labels and saved warning metrics."""
import json
import platform

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

from scripts.onset_research import OUT, ROOT, digest, daily_discharge, verify_protocol, write_json
from scripts.onset_worker import predict


def main():
    verify_protocol()
    provenance = json.loads((OUT / 'provenance.json').read_text())
    selection = json.loads((OUT / 'selection.json').read_text())
    evaluation = json.loads((OUT / 'evaluation.json').read_text())
    assert evaluation['selection_sha256'] == digest(OUT / 'selection.json')
    assert selection['provenance_sha256'] == digest(OUT / 'provenance.json')
    for path, checksum in selection['code'].items():
        assert digest(ROOT / path) == checksum, path
    for name, record in provenance['datasets'].items():
        assert digest(OUT / f'{name}.csv.gz') == record['sha256'], name
    test = pd.read_csv(OUT / 'test.csv.gz', dtype={'station': str, 'huc': str}).fillna({'event_date': ''})
    registry = {s['station']: s for s in json.loads((OUT / 'stations.json').read_text())}
    checked = 0
    for sid, group in test.groupby('station'):
        source = next(ROOT / path for path in provenance['sources'] if path.endswith('/' + sid + '.rdb'))
        assert digest(source) == provenance['sources'][str(source.relative_to(ROOT))]['sha256']
        q = daily_discharge(source.read_text(), sid)
        q.index = pd.to_datetime(q.index)
        reference = q[q.index < '2016-01-01'].quantile(.95)
        for row in group.itertuples():
            issue = pd.Timestamp(row.decision_date)
            quiet = q.reindex(pd.date_range(issue - pd.Timedelta(days=9), periods=7))
            assert quiet.notna().all() and quiet.le(reference).all()
            complete = q.reindex(pd.date_range(issue - pd.Timedelta(days=6), periods=10))
            assert complete.notna().all()
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
            checked += 1
    for candidate, record in zip(selection['candidates'], evaluation['candidates']):
        name = candidate['name']
        artifact = OUT / 'artifacts' / f'{name}.joblib'
        assert digest(artifact) == candidate['artifact_sha256']
        bundle = joblib.load(artifact)
        saved = pd.read_csv(OUT / 'predictions' / f'{name}.test.csv.gz', dtype={'station': str, 'huc': str}).fillna({'event_date': ''})
        assert saved[['station', 'decision_date', 'event_date']].equals(test[['station', 'decision_date', 'event_date']])
        p = saved.probability.to_numpy(); y = saved.exceedance.to_numpy(bool)
        assert np.allclose(predict(bundle, test), p, rtol=0, atol=1e-14)
        overall = record['overall']
        assert np.isclose(average_precision_score(y, p), overall['ap'])
        assert np.isclose(brier_score_loss(y, p), overall['brier'])
        for policy, metrics in overall['policies'].items():
            alerts = saved[policy].to_numpy(bool)
            events = saved[y][['station', 'event_date']].drop_duplicates()
            hits = saved[y & alerts][['station', 'event_date']].drop_duplicates()
            assert len(events) == metrics['events']
            assert np.isclose(len(hits) / len(events), metrics['event_recall'])
            assert int(np.sum(alerts & ~y)) == metrics['false_alerts']
    write_json(OUT / 'audit.json', dict(label_rows_reconstructed=checked, candidates_reproduced=len(evaluation['candidates']),
        evaluation_sha256=digest(OUT / 'evaluation.json'), audit_code_sha256=digest(ROOT / 'scripts/audit_onset.py'),
        platform=platform.platform(), machine=platform.machine(), python=platform.python_version(),
        production_preserved=True, status='passed'))
    print('Audited', checked, 'labels and all saved candidate predictions')


if __name__ == '__main__':
    main()

"""Reconstruct measured outcomes and replay frozen local calibration artifacts."""
import argparse
import json

import joblib
import numpy as np
import pandas as pd
from shapely.geometry import shape

from scripts.localcal_research import ROOT, OUT, digest, write_json, load_data, adjusted, fit_intercept
from scripts.localcal_worker import verify_selection, METHODS, local_probabilities
from scripts.analog_research import OUT as ANALOG
from scripts.analog_worker import predict, verify_selection as verify_analog
from scripts.hydrology_research import daily_discharge
from scripts.onset_worker import metrics


def audit(family):
    selection = verify_selection(); old = verify_analog()
    cal,test = load_data('calibration'),load_data('test')
    sources = {}
    for split in ['calibration','test']:
        provenance = json.loads((OUT/f'{split}.provenance.json').read_text())
        assert provenance['protocol_sha256'] == digest(OUT/'protocol.json')
        for path,record in provenance['sources'].items():
            assert digest(ROOT/path) == record['sha256']
            assert (ROOT/path).stat().st_size == record['bytes']
            assert json.loads((ROOT/path).with_name((ROOT/path).name+'.request.json').read_text()) == record
            sources[path] = record
    cohort = json.loads((OUT/'cohort.json').read_text())
    registry = {s['station']:s for s in json.loads((OUT/'stations.json').read_text())}
    assert set(test.station) <= set(registry) <= set(selection['calibration']) | {s['station'] for s in json.loads((OUT/'calibration.provenance.json').read_text())['coverage'] if 'excluded' in s}
    assert not set(registry) & set(cohort['prior_station_provenance'])
    occupied = []
    for entry in cohort['occupied_geometry_hashes']:
        assert digest(ROOT/entry['path']) == entry['sha256']
        occupied.append(shape(json.loads((ROOT/entry['path']).read_text())['features'][0]['geometry']))
    for entry in cohort['selected']:
        assert digest(ROOT/entry['basin']['path']) == entry['basin']['sha256']
        polygon = shape(json.loads((ROOT/entry['basin']['path']).read_text())['features'][0]['geometry'])
        assert polygon.is_valid and all(polygon.intersection(other).area == 0 for other in occupied)
        occupied.append(polygon)
    if family == 'classic':
        for sid,frame in test.groupby('station'):
            path = next(ROOT/p for p in sources if p.endswith('/sources/'+sid+'.rdb'))
            q = daily_discharge(path.read_text(),sid); q.index = pd.to_datetime(q.index)
            calibration_path = next(ROOT/p for p in sources if p.endswith('/calibration-sources/'+sid+'.rdb'))
            earlier = daily_discharge(calibration_path.read_text(),sid); earlier.index = pd.to_datetime(earlier.index)
            pd.testing.assert_series_equal(q.loc[:'2021-12-31'],earlier)
            threshold = q[q.index<'2016-01-01'].quantile(.95)
            assert np.allclose(frame.reference_cfs,threshold)
            issue = pd.DatetimeIndex(frame.decision_date)
            history = np.column_stack([q.reindex(issue+pd.Timedelta(days=d)).to_numpy() for d in range(-9,4)])
            assert np.isfinite(history).all() and (history[:,:7] <= threshold).all()
            event = np.full(len(frame),'',dtype=object)
            for lead in [3,2,1]:
                mask = (history[:,9+lead]>threshold) & (history[:,9+lead-7:9+lead]<=threshold).all(axis=1)
                event[mask] = (issue[mask]+pd.Timedelta(days=lead)).strftime('%Y-%m-%d')
            assert np.array_equal(event,frame.event_date) and np.array_equal(event!='',frame.exceedance.astype(bool))
            positive = event!=''
            local = pd.DatetimeIndex(event[positive]).tz_localize(registry[sid]['timezone']).tz_convert('UTC').tz_localize(None)
            assert np.allclose((local-issue[positive]).total_seconds()/3600,frame.loc[positive,'event_lead_hours'])
    combined = json.loads((OUT/'evaluation.json').read_text())
    assert combined['evaluation_hashes'][family] == digest(OUT/f'{family}.evaluation.json')
    assert combined['selection_sha256'] == digest(OUT/'selection.json')
    assert combined['test_sha256'] == digest(OUT/'test.csv.gz')
    evaluation = json.loads((OUT/f'{family}.evaluation.json').read_text())
    assert evaluation['selection_sha256'] == digest(OUT/'selection.json')
    assert evaluation['test_provenance_sha256'] == digest(OUT/'test.provenance.json')
    baseline = joblib.load(ANALOG/'artifacts/offset.joblib') if family=='classic' else None
    global_p = predict(baseline,test) if baseline else None
    if family == 'classic':
        pcal = predict(baseline,cal)
        for name,penalty in METHODS.items():
            for sid,g in cal.groupby('station'):
                assert np.isclose(fit_intercept(pcal[g.index],g.exceedance,penalty),selection['intercepts'][name][sid],atol=1e-10)
    replayed = {}
    for row in evaluation['candidates']:
        name = row['name']; path = OUT/'predictions'/f'{name}.test.csv.gz'
        saved = pd.read_csv(path,dtype={'station':str,'huc':str},float_precision='round_trip').fillna({'event_date':''})
        pd.testing.assert_frame_equal(saved[test.columns.intersection(saved.columns)],test[test.columns.intersection(saved.columns)])
        if name == 'global' or name in METHODS:
            p = global_p if name=='global' else local_probabilities(test,global_p,selection['intercepts'][name])
            cuts = next(r['thresholds'] for r in selection['candidates'] if r['name']==name)
            for _,g in test.groupby('station'):
                assert np.array_equal(np.argsort(p[g.index],kind='stable'),np.argsort(global_p[g.index],kind='stable'))
        else:
            r = next(r for r in old['candidates'] if r['name']==name)
            p = predict(joblib.load(ROOT/r['artifact']),test); cuts=r['thresholds']
        assert np.array_equal(p,saved.probability.to_numpy())
        for key,cut in cuts.items():
            assert np.array_equal(saved[key],p>=cut)
        actual = metrics(test,p,{k:saved[k].to_numpy() for k in row['overall']['policies']})
        assert actual == row['overall']
        replayed[name] = digest(path)
    write_json(OUT/f'{family}.audit.json',dict(status='passed',source_files=len(sources),
        new_source_bytes=sum(r['bytes'] for p,r in sources.items() if p.startswith(str(OUT.relative_to(ROOT)))),
        test_rows=len(test),test_gauges=int(test.station.nunique()),occupied_basins=len(cohort['occupied_geometry_hashes']),
        reconstructed_labels=len(test) if family=='classic' else 0,predictions=replayed,
        evaluation_sha256=digest(OUT/f'{family}.evaluation.json'),code_sha256=digest(ROOT/'scripts/audit_localcal.py')))
    print(family,'audit passed: sources, calibration, separation, frozen prediction replay',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('family',choices=['classic','neural'])
    audit(parser.parse_args().family)

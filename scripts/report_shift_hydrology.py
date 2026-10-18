"""Verify frozen predictions, event coverage and precommitted promotion gates."""
import json

import pandas as pd

from scripts.shift_hydrology import OUT, protocol, digest, write_json, verify_selection
from scripts.shift_hydrology_metrics import metrics, paired_uncertainty, jointly_detected_lead_delta


def promotion_gates(evidence):
    gates = {}
    for partition in ['temporal_test','geographic_test']:
        pair = evidence[partition]
        c,b = pair['candidate'],pair['baseline']
        ci = pair['uncertainty']['ap_difference_95']
        gates[partition+'_events'] = c['events'] >= 20
        gates[partition+'_ap'] = ci is not None and ci[0] > 0
        gates[partition+'_event_gain'] = (c['event_recall'] is not None and b['event_recall'] is not None
                                         and c['event_recall']-b['event_recall'] >= .05-1e-12)
        gates[partition+'_brier'] = c['brier'] <= b['brier']
        gates[partition+'_regions'] = all(r['false_alerts_per_100'] <= 3 for r in c['regions'].values())
        lead = pair['lead']['median_lead_delta_hours']
        gates[partition+'_joint_lead'] = lead is not None and lead >= 0
    pooled = evidence['pooled']
    gates['total_events'] = pooled['candidate']['events'] >= 50
    gates['pooled_false_alerts'] = pooled['candidate']['false_alerts_per_100'] <= 1.5
    ci = pooled['uncertainty']['event_recall_difference_95']
    gates['pooled_event_uncertainty'] = ci is not None and ci[0] > 0
    return dict(gates=gates,statistical_pass=all(gates.values()),integration_ready=False,
        operational_blockers=['No historical telemetry receipt/revision evidence',
                              'No verified construction-site gauge applicability',
                              'High-flow percentile target is not a verified flooding threshold'])


def read_predictions(record, partition, evaluations, expected_labels):
    path = OUT/'predictions'/(record['name']+'-'+partition+'.csv.gz')
    expected = next(r['prediction_sha256'] for r in evaluations
                    if r['name']==record['name'] and r['partition']==partition)
    if digest(path) != expected:
        raise ValueError('Saved prediction checksum changed')
    saved=pd.read_csv(path,dtype={'station':str,'huc':str},keep_default_na=False)
    pd.testing.assert_frame_equal(saved[expected_labels.columns].reset_index(drop=True),
                                  expected_labels.reset_index(drop=True))
    return saved


def paired_frame(control,candidate):
    columns = ['station','huc','decision_time','horizon','target','event_time']
    pd.testing.assert_frame_equal(control[columns],candidate[columns])
    frame = control[columns].copy()
    frame['control'] = control.probability.to_numpy()
    frame['candidate'] = candidate.probability.to_numpy()
    return frame


def coverage(frame, ledger, horizon, role):
    represented = frame.loc[frame.target.eq(1),['station','event_time']].drop_duplicates()
    represented_pairs = set(map(tuple,represented.to_numpy()))
    records = [r for r in ledger if r['role']==role]
    all_events = {(r['station'],event) for r in records for event in r.get('events',[])}
    from collections import Counter
    losses = Counter(event['reasons'][str(horizon)] for r in records for event in r.get('event_eligibility',[]))
    expected = {(r['station'],event['event_time']) for r in records for event in r.get('event_eligibility',[])
                if event['reasons'][str(horizon)]=='eligible'}
    if expected != represented_pairs:
        raise ValueError('Event eligibility diagnostics disagree with evaluated labels')
    if not represented_pairs.issubset(all_events):
        raise ValueError('Positive prediction labels not found in raw event ledger')
    blind = 0
    for station,group in frame.groupby('station'):
        decisions = pd.to_datetime(group.decision_time,utc=True)
        event_times = pd.DatetimeIndex([event for sid,event in all_events if sid==station])
        for event in event_times:
            blind += int(((decisions>=event)&(decisions<event+pd.Timedelta(hours=2))).sum())
    return dict(horizon=horizon,all_certified_onsets=len(all_events),eligible_events=len(represented_pairs),
        eligibility_reasons=dict(losses),excluded_gauges=[r['station'] for r in records if 'excluded' in r],
        events_without_eligible_warning_decision=len(all_events-represented_pairs),
        eligible_decisions_with_onset_in_observation_blind_interval=blind,
        event_coverage=len(represented_pairs)/len(all_events) if all_events else None)


def report():
    from scripts.shift_hydrology_worker import load_data,partitions
    protocol()
    if (OUT/'evaluation.json').exists():
        raise FileExistsError('Final report already recorded')
    selected=verify_selection()
    records={r['name']:r for r in selected['candidates']}
    labels=['station','huc','decision_time','horizon','target','event_time']
    canonical={key:value[labels] for key,value in partitions(load_data('test')).items()}
    evaluations=[]
    summary=[]
    for family in ['baseline','logistic','lightgbm','xgboost']:
        family_result=json.loads((OUT/(family+'-evaluation.json')).read_text())
        if family_result['selection_sha256'] != digest(OUT/'selection.json'):
            raise ValueError('Evaluation used a different selection')
        if family_result['dataset_sha256'] != digest(OUT/'test.csv.gz'):
            raise ValueError('Evaluation used a different test dataset')
        for r in family_result['results']:
            evaluations.append(r)
            expected=canonical[r['partition']]
            read_predictions(records[r['name']],r['partition'],evaluations,
                             expected.loc[expected.horizon.eq(records[r['name']]['horizon'])])
            s=dict(name=r['name'],partition=r['partition'])
            s.update({key:r['metrics'][key] for key in ['rows','positives','ap','brier','events','detected_events',
                'event_recall','median_lead_hours','false_alerts_per_100','false_warning_episodes']})
            summary.append(s)
    pd.DataFrame(summary).to_csv(OUT/'summary.csv',index=False)
    ledger=json.loads((OUT/'test-coverage.json').read_text())
    results={}
    for horizon in [6,12,24]:
        pair=selected['selected'][str(horizon)]
        baseline,candidate=records[pair['baseline']],records[pair['candidate']]
        evidence={}
        frames=[]
        for partition in ['temporal_test','geographic_test']:
            expected=canonical[partition].loc[canonical[partition].horizon.eq(horizon)]
            control=read_predictions(baseline,partition,evaluations,expected)
            test=read_predictions(candidate,partition,evaluations,expected)
            f=paired_frame(control,test)
            frames.append(f)
            evidence[partition]=dict(
                baseline=metrics(f,f.control,baseline['threshold']),
                candidate=metrics(f,f.candidate,candidate['threshold']),
                uncertainty=paired_uncertainty(f,f.control,f.candidate,baseline['threshold'],candidate['threshold']),
                lead=jointly_detected_lead_delta(f,f.control,f.candidate,baseline['threshold'],candidate['threshold']),
                coverage=coverage(f,ledger,horizon,'development' if partition=='temporal_test' else 'geographic_test'),
                stations={sid:dict(baseline=metrics(g,g.control,baseline['threshold']),
                                   candidate=metrics(g,g.candidate,candidate['threshold'])) for sid,g in f.groupby('station')})
        pooled=pd.concat(frames,ignore_index=True)
        evidence['pooled']=dict(baseline=metrics(pooled,pooled.control,baseline['threshold']),
            candidate=metrics(pooled,pooled.candidate,candidate['threshold']),
            uncertainty=paired_uncertainty(pooled,pooled.control,pooled.candidate,baseline['threshold'],candidate['threshold']))
        evidence.update(selection=pair,promotion=promotion_gates(evidence))
        results[str(horizon)]=evidence
        print(horizon,pair,'statistical pass',evidence['promotion']['statistical_pass'],flush=True)
    write_json(OUT/'evaluation.json',dict(horizons=results,
        selection_sha256=digest(OUT/'selection.json'), test_sha256=digest(OUT/'test.csv.gz')))


if __name__=='__main__':
    report()

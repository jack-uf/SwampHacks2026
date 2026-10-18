from copy import deepcopy

from scripts.report_shift_hydrology import promotion_gates


def evidence():
    pair = dict(candidate=dict(events=30,event_recall=.7,brier=.01,regions={'01':{'false_alerts_per_100':1}}),
        baseline=dict(events=30,event_recall=.6,brier=.02),
        uncertainty=dict(ap_difference_95=[.01,.05]),
        lead=dict(jointly_detected_events=15,median_lead_delta_hours=0))
    return dict(temporal_test=deepcopy(pair),geographic_test=deepcopy(pair),
        pooled=dict(candidate=dict(events=60,false_alerts_per_100=1),
                    uncertainty=dict(event_recall_difference_95=[.01,.1])))


def test_every_gate_is_required_and_retrospective_data_cannot_authorize_integration():
    result = promotion_gates(evidence())
    assert result['statistical_pass']
    assert not result['integration_ready']
    changed = evidence()
    changed['geographic_test']['candidate']['regions']['17'] = {'false_alerts_per_100':3.01}
    assert not promotion_gates(changed)['statistical_pass']
    changed = evidence()
    changed['pooled']['uncertainty']['event_recall_difference_95'] = [0,.1]
    assert not promotion_gates(changed)['statistical_pass']

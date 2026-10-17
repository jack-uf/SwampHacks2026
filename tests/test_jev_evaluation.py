"""Offline checks of metric arithmetic; these are not live model measurements."""
import copy
import json
from scripts.evaluate_jev import metrics, plan, PROTOCOL


def test_frozen_protocol_matches_current_questions_and_labels():
    cases,configuration=plan()
    assert configuration==json.loads(PROTOCOL.read_text())
    assert len(cases)==10


def test_metrics_count_multilabel_errors_abstentions_and_provider_failure():
    row=dict(id='fixture',expected=dict(categories=['flood','wildlife'],routing='hazard_review',urgency=1),
             result=dict(status='classified',categories=['flood','heat'],routing=dict(choice='hazard_review'),
                         urgency=dict(score=1.5),abstained=True,uncertain=['wildlife'],review_queue='clarification'),latency_seconds=1.)
    failed=copy.deepcopy(row);failed['id']='outage';failed['result']=dict(status='unavailable')
    result=metrics([row,failed])
    assert result['micro']==dict(tp=1,fp=1,fn=1,precision=.5,recall=.5)
    assert result['response_coverage']==.5 and result['failed']==1
    assert result['abstention_rate']==1 and result['uncertain_rate']==1
    assert result['urgency_mean_absolute_error']==.5 and len(result['disagreements'])==1
    assert result['per_class']['storm']['recall'] is None
    assert metrics([failed])['micro']['precision'] is None

"""Offline checks of metric arithmetic; these are not live model measurements."""
import copy
import json
from scripts.evaluate_jev import abstention_comparison, metrics, plan, PROTOCOL, SUITES


def test_frozen_protocol_matches_current_questions_and_labels():
    cases,configuration=plan()
    assert configuration==json.loads(PROTOCOL.read_text())
    assert len(cases)==10


def test_abstention_comparison_scores_both_rules_on_the_same_responses():
    from test_jev import payload
    from backend import jev
    def row(id, expected, probabilities):
        return dict(id=id, expected=dict(categories=expected, routing='hazard_review', urgency=1),
                    result=jev.interpret(payload(probabilities)), latency_seconds=.1)
    rows = [row('none', [], {}), row('right', ['flood'], {'flood': .9}), row('wrong', ['heat'], {'flood': .9}),
            row('unsure', ['flood'], {'flood': .9, 'heat': .5})]
    result = abstention_comparison(rows)
    assert result['v1'] == dict(rule='v1', responses=4, abstained=2, abstention_rate=.5, decisive=2, decisive_exact=1,
                                decisive_exact_rate=.5, decisive_wrong=1, abstained_exact=2)
    assert result['v2'] == dict(rule='hazard-decisive-v2', responses=4, abstained=1, abstention_rate=.25, decisive=3, decisive_exact=2,
                                decisive_exact_rate=2/3, decisive_wrong=1, abstained_exact=1)
    assert result['review_queue_changed'] == 0


def test_heldout_protocol_is_frozen_with_rule_source():
    cases, configuration = plan('heldout')
    assert configuration == json.loads(SUITES['heldout']['protocol'].read_text())
    assert len(cases) == configuration['scenarios'] == 16
    assert not {c['text'] for c in cases} & {c['text'] for c in plan()[0]}


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

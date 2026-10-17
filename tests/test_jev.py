"""Synthetic response fixtures verify the workflow, not Jev's semantic accuracy."""
import asyncio
import copy
import json
import httpx
import pytest
from backend import jev
from scripts.common import ROOT


def payload(probabilities=None):
    values={key:.05 for key in jev.INSTRUCTIONS};values.update(probabilities or {})
    answers={key:dict(type='noul',noul=value) for key,value in values.items()}
    answers['routing']=dict(type='choice',choice='hazard_review',probabilities=dict(hazard_review=1.,clarify_report=0.,routine_log=0.),confidence=1.)
    answers['urgency']=dict(type='score',score=1.,legend={str(i):v for i,v in enumerate(jev.URGENCY)},probabilities={'0':0.,'1':1.,'2':0.},confidence=1.)
    return dict(model='jev-1.13.0',answers=answers,usage=dict(input_tokens=150,output_tokens=12))


def test_hypothetical_triage_workflow_fixtures():
    examples=json.loads((ROOT/'tests/fixtures/jev-hypothetical.json').read_text())
    for example in examples:
        result=jev.interpret(payload(example['fixture_probabilities']))
        assert result['categories']==example['expected_categories'],example['id']
        assert result['abstained']==example['expected_abstention']
        assert result['abstained_v1']==example['expected_abstention_v1']
        assert result['needs_clarification']==example['expected_clarification']
        assert result['reviewed'] is False


@pytest.mark.parametrize('change',[
    lambda p:p['answers'].pop('heat'),lambda p:p['answers']['heat'].update(noul=1.1),
    lambda p:p['answers']['heat'].update(noul=float('nan')),lambda p:p['answers']['heat'].update(noul='0.8'),
    lambda p:p['answers']['heat'].update(type='choice'),lambda p:p['usage'].update(input_tokens=-1),
    lambda p:p['usage'].update(output_tokens=True),lambda p:p['answers']['routing'].update(choice='safe'),
    lambda p:p['answers']['routing']['probabilities'].update(routine_log=.8),
    lambda p:p['answers']['routing'].update(choice='routine_log'),
    lambda p:p['answers']['urgency']['legend'].update({'0':'Safe to work'}),
    lambda p:p['answers']['urgency'].update(score=2.),
    lambda p:p['answers']['urgency']['probabilities'].update({'0':float('inf')}),
    lambda p:p['answers']['urgency'].update(confidence='0.9')])
def test_malformed_provider_contract_rejected(change):
    p=payload();change(p)
    with pytest.raises(ValueError):jev.interpret(p)


def test_external_calls_require_consent_enabled_and_key(monkeypatch):
    def forbidden(*a,**k):raise AssertionError('No external client may be created')
    monkeypatch.setattr(jev.httpx,'AsyncClient',forbidden)
    monkeypatch.setenv('PLANIT_JEV_ENABLED','true');monkeypatch.setenv('TYPESAFE_API_KEY','fixture-secret')
    assert asyncio.run(jev.classify('A gator by road.',consent=False))['status']=='unavailable'
    monkeypatch.setenv('PLANIT_JEV_ENABLED','false')
    assert asyncio.run(jev.classify('A gator by road.',consent=True))['status']=='unavailable'
    monkeypatch.setenv('PLANIT_JEV_ENABLED','true');monkeypatch.delenv('TYPESAFE_API_KEY')
    assert asyncio.run(jev.classify('A gator by road.',consent=True))['status']=='unavailable'


@pytest.mark.parametrize('mode',['ok','401','403','429','529','timeout','network','malformed','model'])
def test_provider_failure_is_manual_and_text_only(monkeypatch,mode):
    monkeypatch.setenv('PLANIT_JEV_ENABLED','true');monkeypatch.setenv('TYPESAFE_API_KEY','fixture-secret');monkeypatch.setenv('PLANIT_JEV_MODEL','jev-1.13.0')
    calls=[]
    class Client:
        def __init__(self,**kwargs):assert kwargs['timeout']==15
        async def __aenter__(self):return self
        async def __aexit__(self,*args):pass
        async def post(self,url,**kwargs):
            calls.append(kwargs)
            assert url=='https://api.typesafe.ai/v1/systemone'
            if mode=='timeout':raise httpx.ReadTimeout('fixture')
            if mode=='network':raise httpx.ConnectError('fixture')
            p=payload({'wildlife':.9})
            if mode=='malformed':p['answers']={}
            if mode=='model':p['model']='wrong-model'
            return httpx.Response(int(mode) if mode.isdigit() else 200,json=p,headers={'Retry-After':'30'})
    monkeypatch.setattr(jev.httpx,'AsyncClient',Client)
    result=asyncio.run(jev.classify('Gator beside the east access road.',consent=True))
    assert len(calls)==1
    assert calls[0]['json']['state']=={'observation':'Gator beside the east access road.'}
    assert set(calls[0]['json'])=={'model','state','questions'}
    assert result['status']==('classified' if mode=='ok' else 'unavailable')
    assert 'fixture-secret' not in json.dumps(result)


def test_confident_no_hazard_is_decisive_but_still_unreviewed():
    result=jev.interpret(payload({'location_sufficient':.9}))
    assert result['categories']==[] and result['abstained'] is False and result['abstained_v1'] is True
    assert result['abstention']['reasons']==[] and result['reviewed'] is False


def test_auxiliary_uncertainty_keeps_clarification_queue_without_abstaining():
    result=jev.interpret(payload({'flood':.95,'location_sufficient':.5,'needs_clarification':.5}))
    assert result['abstained'] is False and result['review_queue']=='clarification' and result['needs_clarification']


def test_uncertain_specific_hazard_or_unresolved_other_abstains():
    result=jev.interpret(payload({'flood':.95,'wildlife':.5}))
    assert result['abstained'] and result['abstention']['reasons']==['wildlife uncertain']
    assert jev.interpret(payload({'other':.6}))['abstention']['reasons']==['other uncertain and no specific hazard accepted']
    flagged=jev.interpret(payload({'lightning':.98,'other':.6}))
    assert flagged['abstained'] is False and flagged['abstention']['possible_other'] is True


def test_v2_rule_reproduces_stored_live_responses_and_changes_only_abstention():
    rows=[json.loads(line) for line in (ROOT/'output/jev-evaluation.jsonl').read_text().splitlines()]
    for row in [r for r in rows if r['kind']=='response']:
        stored=row['result']
        result=jev.interpret(dict(model=stored['model'],answers=stored['answers'],usage=stored['usage']))
        assert result['abstained_v1']==stored['abstained'] and result['review_queue']==stored['review_queue']
        assert result['categories']==stored['categories'] and result['uncertain']==stored['uncertain']


def test_typed_routing_uncertainty_and_urgency():
    p=payload({'flood':.98,'unstable_terrain':.9})
    p['answers']['routing'].update(confidence=.4)
    result=jev.interpret(p)
    assert result['review_queue']=='clarification' and result['abstained']
    assert result['uncertain']==['routing']
    p['answers']['urgency'].update(score=2.,probabilities={'0':0.,'1':0.,'2':1.})
    result=jev.interpret(p)
    assert result['review_queue']=='urgent' and result['reviewed'] is False
    assert result['categories']==['flood','unstable_terrain']
    assert result['answers']['urgency']['legend']['2']==jev.URGENCY[2]
    p['answers']['routing'].update(choice='clarify_report',confidence=1.,probabilities=dict(hazard_review=0.,clarify_report=1.,routine_log=0.))
    assert jev.interpret(p)['needs_clarification']

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
    return dict(model='jev-1.13.0',answers={key:dict(type='noul',noul=value) for key,value in values.items()},usage=dict(input_tokens=150,output_tokens=12))


def test_hypothetical_triage_workflow_fixtures():
    examples=json.loads((ROOT/'tests/fixtures/jev-hypothetical.json').read_text())
    for example in examples:
        result=jev.interpret(payload(example['fixture_probabilities']))
        assert result['categories']==example['expected_categories'],example['id']
        assert result['abstained']==example['expected_abstention']
        assert result['needs_clarification']==example['expected_clarification']
        assert result['reviewed'] is False


@pytest.mark.parametrize('change',[lambda p:p['answers'].pop('heat'),lambda p:p['answers']['heat'].update(noul=1.1),lambda p:p['answers']['heat'].update(noul=float('nan')),lambda p:p['answers']['heat'].update(noul='0.8'),lambda p:p['answers']['heat'].update(type='choice'),lambda p:p['usage'].update(input_tokens=-1)])
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
    if mode=='429':assert result['retry_after']=='30'

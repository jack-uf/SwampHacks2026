import asyncio
from datetime import timedelta
import httpx
from backend import providers
from backend.domain import now_utc
from backend.store import initialize, put_cache, get_cache


def test_failed_refresh_preserves_last_success_and_cools_down(tmp_path, monkeypatch):
    monkeypatch.setenv('PLANIT_DB',str(tmp_path/'providers.sqlite'))
    initialize()
    old=(now_utc()-timedelta(hours=2)).isoformat()
    put_cache('test',dict(data={'original':True},last_success_at=old),old)
    calls=[]
    class FailingClient:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self,*args): pass
        async def get(self,*args,**kwargs):
            calls.append(True)
            raise httpx.ReadTimeout('')
    monkeypatch.setattr(providers.httpx,'AsyncClient',FailingClient)
    first=asyncio.run(providers.fetch('test','https://example.test',{}))
    second=asyncio.run(providers.fetch('test','https://example.test',{}))
    assert not first['available'] and first['stale']
    assert first['fetched_at']==second['fetched_at']==old
    assert first['error'].startswith('ReadTimeout:') and second['data']=={'original':True}
    assert len(calls)==1 and get_cache('test')['error']


def test_unmapped_flood_exposure_is_unknown():
    from backend.domain import hazard_index
    result=hazard_index(None,None,[],{},dict(zones=[],feed=dict(available=True)))
    assert next(c for c in result['components'] if c['name']=='Mapped flood exposure')['severity'] is None

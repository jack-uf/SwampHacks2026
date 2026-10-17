import hashlib
import json
import httpx
import pytest
from scripts import common


def test_cached_checksum_and_rate_limit_stop(tmp_path,monkeypatch):
    monkeypatch.setattr(common,'RAW',tmp_path)
    path=tmp_path/'sample';path.write_bytes(b'original')
    (tmp_path/'sample.source.json').write_text(json.dumps(dict(sha256=hashlib.sha256(b'original').hexdigest())))
    assert common.download('https://example.test','sample')==path
    path.write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='checksum'):common.download('https://example.test','sample')
    calls=[];sleeps=[]
    def get(*args,**kwargs):
        calls.append(True);return httpx.Response(429,headers={'Retry-After':'3'},request=httpx.Request('GET','https://example.test'))
    monkeypatch.setattr(common.httpx,'get',get);monkeypatch.setattr(common.time,'sleep',sleeps.append)
    with pytest.raises(RuntimeError,match='Repeated'):common.download('https://example.test','new')
    assert len(calls)==2 and sleeps==[3]

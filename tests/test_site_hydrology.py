"""Synthetic boundaries plus integration checks for river context; no synthetic live evidence."""
import asyncio
import hashlib
import json
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from backend import hydrology, main
from backend.domain import now_utc, parse_time
from backend.store import db, initialize, put_cache
from tests.test_api import client, AUTH


def payload(station='04127997', quality='Approved', start='2024-02-01T00:00:00+00:00', periods=193):
    dates = pd.date_range(start, periods=periods, freq='h')
    return dict(type='FeatureCollection', links=[], features=[dict(properties=dict(
        monitoring_location_id='USGS-'+station, parameter_code='00060', statistic_id='00011',
        unit_of_measure='ft^3/s', time_series_id='one', time=t.isoformat(),
        value=str(40 + i*.05), approval_status=quality, qualifier=None)) for i, t in enumerate(dates)])


def test_parser_rejects_ambiguity_wrong_units_pagination_and_future():
    received = parse_time('2024-03-01T00:00:00+00:00')
    good = payload(periods=4)
    assert len(hydrology.parse_discharge(good, '04127997', received)) == 4
    for mutate in [lambda x: x['features'][0]['properties'].update(unit_of_measure='m3/s'),
                   lambda x: x['features'][0]['properties'].update(monitoring_location_id='USGS-00000000'),
                   lambda x: x['features'][0]['properties'].update(time_series_id='two'),
                   lambda x: x['features'].append(x['features'][0]),
                   lambda x: x.update(links=[dict(rel='next', href='https://example.org')]),
                   lambda x: x['features'][0]['properties'].update(time='2025-01-01T00:00:00Z')]:
        changed = json.loads(json.dumps(good))
        mutate(changed)
        with pytest.raises(ValueError):
            hydrology.parse_discharge(changed, '04127997', received)


def test_live_features_match_the_frozen_research_contract_without_future_readings():
    from scripts.shift_hydrology import decision_rows, hourly_flow
    received = parse_time('2024-02-09T00:00:00+00:00')
    rows = hydrology.parse_discharge(payload(), '04127997', received)
    station = dict(latitude=45., longitude=-84., elevation_m=232., area_km2=497., reference_cfs=100.,
                   station='04127997', huc='04070004', role='development')
    live = hydrology.feature_row(rows, station, received)
    q = pd.Series([r['value_cfs'] for r in rows], index=pd.to_datetime([r['time'] for r in rows]))
    q = q.reindex(pd.date_range(q.index[0], q.index[-1]+pd.Timedelta(hours=48), freq='h')).ffill()
    frozen, features, _ = decision_rows(hourly_flow(q), 100, station)
    expected = frozen.loc[(frozen.horizon==6) & frozen.decision_time.eq(received.isoformat()), features]
    np.testing.assert_allclose(live[features].to_numpy(), expected.to_numpy(), atol=1e-15)
    altered = [dict(r, value_cfs=99999) if parse_time(r['time']) > received-timedelta(hours=2) else r for r in rows]
    pd.testing.assert_frame_equal(live, hydrology.feature_row(altered, station, received))


def test_provisional_and_missing_history_are_withheld_but_observations_remain_visible():
    issue = parse_time('2024-02-09T00:00:00+00:00')
    rows = hydrology.parse_discharge(payload(quality='Provisional'), '04127997', issue)
    obs = hydrology.observation_summary(rows, issue, issue)
    assert obs['available'] and obs['quality'] == 'Provisional' and obs['approved_readings'] == 0
    assert obs['observation_to_receipt_minutes'] == 0
    with pytest.raises(ValueError, match='approved_history_missing'):
        hydrology.feature_row(rows, {}, issue)
    rows = hydrology.parse_discharge(payload(), '04127997', issue)
    with pytest.raises(ValueError, match='incomplete'):
        hydrology.feature_row(rows[:100]+rows[101:], {'reference_cfs': 100}, issue)
    with pytest.raises(ValueError, match='not_quiet'):
        hydrology.feature_row(rows, {'reference_cfs': 10}, issue)
    assert hydrology.observation_summary(rows, issue, issue+timedelta(hours=3))['available'] is False


def test_shadow_uses_receipt_time_and_replays_pinned_models():
    issue = parse_time('2024-02-09T00:00:00+00:00')
    station = '04127997'
    reference = hydrology.manifest()['stations'][station]['reference_cfs']
    x = payload(station)
    for feature in x['features']:
        feature['properties']['value'] = str(reference*.5)
    body = json.dumps(x).encode()
    receipt = dict(id='synthetic-test', body=body, sha256=hashlib.sha256(body).hexdigest(),
        received_at=(issue-timedelta(minutes=10)).isoformat(), status=200)
    # Last synthetic reading arrives after that receipt, so remove it before scoring.
    x['features'] = x['features'][:-1]
    receipt['body'] = json.dumps(x).encode()
    receipt['sha256'] = hashlib.sha256(receipt['body']).hexdigest()
    result = hydrology.shadow(station, receipt, issue)
    assert result['status'] == 'experimental_gauge_score', result
    assert result['operational_status'] == 'withheld'
    assert 0 <= result['probability'] <= 1 and 0 <= result['baseline_probability'] <= 1
    future = hydrology.shadow(station, dict(receipt, received_at=(issue+timedelta(seconds=1)).isoformat()), issue)
    assert future['probability'] is None and 'receipt_outside_issue_availability_window' in future['reasons']
    assert hydrology.shadow(station, None, issue)['probability'] is None
    corrupt = hydrology.shadow(station, dict(receipt, sha256='bad'), issue)
    assert 'receipt_checksum_mismatch' in corrupt['reasons']


def test_receipt_capture_keeps_exact_bytes_and_failure_does_not_reuse_old_success(tmp_path, monkeypatch):
    monkeypatch.setenv('PLANIT_DB', str(tmp_path/'audit.sqlite'))
    initialize()
    body = json.dumps(payload(periods=1)).encode()
    class Client:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): return None
        async def get(self, url, headers):
            return type('Response', (), dict(content=body, status_code=200))()
    monkeypatch.setattr(hydrology.httpx, 'AsyncClient', lambda **kwargs: Client())
    first = asyncio.run(hydrology.capture('04127997'))
    second = asyncio.run(hydrology.capture('04127997'))
    assert first['id'] == second['id']
    with db() as c:
        row = c.execute('SELECT * FROM hydrology_receipts').fetchone()
    assert row['body'] == body and row['sha256'] == hashlib.sha256(body).hexdigest()
    monkeypatch.setattr(hydrology, 'now_utc', lambda: parse_time(first['received_at'])+timedelta(minutes=16))
    async def fail(self, url, headers):
        raise hydrology.httpx.ConnectError('Test offline')
    monkeypatch.setattr(Client, 'get', fail)
    failure = asyncio.run(hydrology.capture('04127997'))
    assert failure['status'] is None and failure['id'] != first['id']
    with db() as c:
        assert c.execute('SELECT count(*) FROM hydrology_receipts').fetchone()[0] == 2


def context(point):
    now = now_utc()
    return dict(status='observed', checked_at=now.isoformat(), stale=False, summary='Synthetic test context',
        latitude=point['latitude'], longitude=point['longitude'], gauges=[dict(station='04127997', name='Synthetic test gauge',
        relationship='upstream_network', receipt=dict(id='receipt-test'), observation=dict(available=True, value_cfs=100,
        quality='Provisional', latest_at=now.isoformat()), experimental=dict(status='withheld', reasons=['model_not_promoted'],
        probability=None, baseline_probability=None, valid_until=(now+timedelta(hours=6)).isoformat()))])


def test_api_connects_point_specific_context_without_changing_ratings_or_window_snapshot(client, monkeypatch):
    from tests.test_mission_api import mission_body
    site = main.SITES[0]
    mid = client.post('/api/missions', json=mission_body(site), headers=AUTH).json()['id']
    before = client.get('/api/missions/'+mid).json()
    original = {e['hazard']: e['rating'] for e in before['jha']['entries']}
    checked = []
    async def inspect(point):
        checked.append(point)
        result = context(point)
        put_cache(hydrology.point_key(point), result, result['checked_at'])
        return result
    monkeypatch.setattr(main.hydrology, 'inspect', inspect)
    assert client.get('/api/missions/'+mid+'/hydrology').status_code == 200
    after = client.get('/api/missions/'+mid).json()
    assert original == {e['hazard']: e['rating'] for e in after['jha']['entries']}
    cited = [e for line in after['jha']['entries'] for e in line['evidence'] if e['id'].startswith('hydrology:')]
    assert cited and all(e['counts'] is False and e['rating'] is None for e in cited)
    assert cited[0]['kind'] == 'observed' and cited[0]['ref'] == 'receipt-test'
    assert after['hydrology']['latitude'] == after['mission']['latitude'] == checked[0]['latitude']
    assert client.get('/api/sites/missing/hydrology').status_code == 404
    assert client.get('/api/missions/missing/hydrology').status_code == 404
    plan = client.get('/api/missions/'+mid+'/work-windows').json()
    assert plan['hydrology']['status'] == 'observed'
    assert main.plan_snapshot(plan) == main.plan_snapshot(dict(plan, hydrology={'probability': 1}))


def test_inventory_filters_inactive_locations_without_conflating_provisional_with_approved(monkeypatch):
    now = now_utc()
    response = payload(station='11111111', quality='Provisional', start=(now-timedelta(minutes=30)).isoformat(), periods=1)
    async def fetch(*args, **kwargs):
        return dict(available=True, data=response, fetched_at=now.isoformat())
    monkeypatch.setattr(hydrology.providers, 'fetch', fetch)
    network = dict(candidates=[dict(station='00000000'), dict(station='11111111')])
    candidates, screen = asyncio.run(hydrology.active_gauges(network, {'latitude': 38., 'longitude': -82.}))
    assert [g['station'] for g in candidates] == ['11111111']
    assert screen['candidates_screened'] == 2 and screen['fresh_discharge_gauges'] == 1
    response['features'][0]['properties']['time'] = (now-timedelta(days=1)).isoformat()
    assert asyncio.run(hydrology.active_gauges(network, {'latitude': 38., 'longitude': -82.}))[0] == []


def test_cached_context_expires_readings_and_forecast_horizon(tmp_path, monkeypatch):
    monkeypatch.setenv('PLANIT_DB', str(tmp_path/'audit.sqlite'))
    initialize()
    point = dict(latitude=38., longitude=-82.)
    result = context(point)
    result['gauges'][0]['observation']['latest_at'] = (now_utc()-timedelta(hours=3)).isoformat()
    result['gauges'][0]['experimental']['valid_until'] = (now_utc()-timedelta(seconds=1)).isoformat()
    put_cache(hydrology.point_key(point), result, result['checked_at'])
    current = hydrology.cached(point)
    assert current['status'] == 'unavailable' and not current['gauges'][0]['observation']['available']
    assert current['gauges'][0]['experimental']['probability'] is None
    assert 'forecast_horizon_expired' in current['gauges'][0]['experimental']['reasons']
    assert hydrology.cached(dict(latitude=38.001, longitude=-82.)) is None

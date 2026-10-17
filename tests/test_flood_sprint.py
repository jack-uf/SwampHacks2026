"""Fictional metadata fixtures test the historical-label trust boundary."""
import hashlib
import json

import pytest

from scripts import flood_sprint
from scripts.flood_sprint import assess_pair


def pair():
    return (
        dict(lid='TEST', usgsId='01234567', latitude=40., longitude=-80.,
             flood=dict(stageUnits='ft', categories=dict(minor=dict(stage=10.))),
             datums=dict(vertical=dict(value=[dict(abbrev='NAVD88', value=100.)]))),
        dict(id='USGS-01234567', properties=dict(site_type_code='ST',
             hydrologic_unit_code='01010101', altitude=100., vertical_datum='NAVD88'),
             geometry=dict(type='Point', coordinates=[-80., 40.])),
    )


def test_current_metadata_does_not_certify_historical_flood_labels():
    result = assess_pair(*pair())
    assert not result['eligible']
    assert result['identity_matches'] and result['distance_km'] == 0
    assert 'historical_threshold_interval_unverified' in result['blockers']
    assert 'historical_gauge_zero_unverified' in result['blockers']


@pytest.mark.parametrize('field,value,blocker', [
    ('usgsId', '99999999', 'station_identifier_mismatch'),
    ('latitude', 41., 'station_locations_disagree'),
])
def test_audit_rejects_inappropriate_station_joins(field, value, blocker):
    gauge, location = pair()
    gauge[field] = value
    assert blocker in assess_pair(gauge, location)['blockers']


def test_same_named_vertical_datum_does_not_erase_elevation_difference():
    gauge, location = pair()
    location['properties']['altitude'] = 99.96
    result = assess_pair(gauge, location)
    assert result['metadata_elevation_difference_ft'] == pytest.approx(.04)
    assert 'current_elevation_difference_requires_review' in result['blockers']


def test_invalid_flood_stage_is_not_a_real_threshold():
    gauge, location = pair()
    gauge['flood']['categories']['minor']['stage'] = -9999
    assert 'minor_flood_stage_unavailable' in assess_pair(gauge, location)['blockers']


def test_missing_vertical_reference_is_explicitly_unresolved():
    gauge, location = pair()
    gauge['datums']['vertical']['value'] = []
    assert 'current_vertical_reference_unresolved' in assess_pair(gauge, location)['blockers']


def test_cached_parse_failure_stays_unavailable_without_another_request(tmp_path, monkeypatch):
    monkeypatch.setattr(flood_sprint, 'OUT', tmp_path)
    source = tmp_path / 'sources'
    source.mkdir()
    payload = b'upstream returned an invalid body'
    (source / 'failed.json').write_bytes(payload)
    (source / 'failed.request.json').write_text(json.dumps(dict(
        url='https://example.test', status=200, error='JSONDecodeError',
        sha256=hashlib.sha256(payload).hexdigest())))
    assert flood_sprint.fetch('failed', 'https://example.test') is None
    assert flood_sprint.cached('failed') is None


def test_offline_audit_rejects_modified_source_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(flood_sprint, 'OUT', tmp_path)
    source = tmp_path / 'sources'
    source.mkdir()
    (source / 'changed.json').write_text('{}')
    (source / 'changed.request.json').write_text(json.dumps(dict(
        status=200, sha256=hashlib.sha256(b'original').hexdigest())))
    with pytest.raises(ValueError, match='Cached response changed'):
        flood_sprint.cached('changed')

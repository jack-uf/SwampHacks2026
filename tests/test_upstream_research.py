import pytest


def test_network_navigation_requires_official_identifiers_and_excludes_self():
    from scripts.upstream_research import upstream_sites
    def feature(sid):
        return dict(type='Feature', properties=dict(identifier='USGS-' + sid, source='nwissite'),
                    geometry=dict(type='Point', coordinates=[-79., 39.]))
    data = dict(type='FeatureCollection', features=[feature('01606500'), feature('01605100')])
    assert upstream_sites(data, '01606500') == ['01605100']
    other = feature('360230093073001')
    other['properties']['identifier'] = 'AR008-360230093073001'
    data['features'].append(other)
    assert upstream_sites(data, '01606500') == ['01605100']
    data['features'].append(feature('01605100'))
    with pytest.raises(ValueError, match='Duplicate'):
        upstream_sites(data, '01606500')
    data['features'] = [feature('bad')]
    with pytest.raises(ValueError, match='identifier'):
        upstream_sites(data, '01606500')


def test_transport_uses_calendar_lags_and_reference_period_only():
    import numpy as np
    import pandas as pd
    from scripts.upstream_pilot import transport_features
    days = pd.date_range('2000-01-01', '2017-01-31')
    up = pd.Series(20 + 5 * np.sin(np.arange(len(days)) / 13), index=days)
    down = up.shift(2).fillna(20) * 3
    issue = pd.DatetimeIndex(['2017-01-10'])
    first, _ = transport_features(down, up, issue)
    expected = np.log1p(up.loc['2017-01-07'] / up.loc[:'2015-12-31'].quantile(.95))
    assert first.iloc[0].up_3 == pytest.approx(expected)
    changed_down = down.copy(); changed_up = up.copy()
    # Alter post-reference outcomes, preserving only the three available predictor days.
    changed_down.loc['2016-01-01':] = 999999
    changed_up.loc['2016-01-01':] = 999999
    for lag in [3, 4, 7]:
        day = issue[0] - pd.Timedelta(days=lag)
        changed_down.loc[day] = down.loc[day]
        changed_up.loc[day] = up.loc[day]
    second, _ = transport_features(changed_down, changed_up, issue)
    pd.testing.assert_frame_equal(first, second)
    missing, _ = transport_features(down, up.drop(pd.Timestamp('2017-01-07')), issue)
    assert missing.isna().all(axis=1).all()


def test_reference_coverage_distinguishes_empty_source_from_unknown_schema():
    from scripts.upstream_pilot import reference_coverage
    empty = '#  No sites found matching all criteria\nagency_cd\tsite_no\tdatetime\n5s\t15s\t20d\n\t\t\n'
    record = reference_coverage(empty, '12357500')
    assert record == dict(station='12357500', eligible=False, approved_days=0,
                         reason='USGS returned no sites for the requested historical interval')
    with pytest.raises(ValueError, match='documented mean discharge'):
        reference_coverage('unrecognized upstream response', '12357500')

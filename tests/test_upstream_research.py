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

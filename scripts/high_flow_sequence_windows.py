"""Hourly window construction for the high-flow sequence experiment.

Shared by the driver and the torch worker. Torch-free on purpose: the driver must never load torch
beside LightGBM in one process, and the worker must never load LightGBM.

Storage stays small because this machine is disk-constrained. The experiment keeps one hourly
four-channel series per gauge (about 700 kilobytes) instead of a materialized window per decision row
(about 2.6 kilobytes each, hundreds of megabytes in total), and every window is sliced from that
series on demand. The series is rebuilt from the cached USGS pages and cached ERA5 responses by the
same functions, so nothing here is a second source of truth.
"""
import gzip
import json
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import pandas as pd

from scripts.hydrology_research import ROOT, digest
from scripts.shift_hydrology import API, hourly_flow, parse_observations

SIGNALS = ROOT/'experiments/high-flow-signals-20261018'
COHORTS = ['shift-hydrology', 'mlp-onset', 'mlp-hybrid', 'mlp-geo']
YEARS = [2020, 2021, 2022, 2023, 2024]
CHANNELS = ['log_flow', 'precipitation', 'temperature_2m', 'soil_moisture_0_to_7cm']
WEATHER_VARIABLES = ['precipitation', 'temperature_2m', 'soil_moisture_0_to_7cm']
WINDOW = 168
WEATHER_DELAY = 6


def intervals(year):
    """Quarterly request intervals, verbatim from the cached-page contract of the earlier sprints."""
    return [(f'{year}-{month:02d}-01T00:00:00Z',
             f'{year + int(month == 10)}-{(month + 3 - 1) % 12 + 1:02d}-01T00:00:00Z')
            for month in [1, 4, 7, 10]]


def cohort_directory(station):
    found = [name for name in COHORTS
             if (ROOT/'experiments'/f'sprint-20261017-{name}'/'sources'/station).is_dir()]
    if len(found) != 1:
        raise ValueError(f'{station} must belong to exactly one cached cohort, found {found}')
    return ROOT/'experiments'/f'sprint-20261017-{found[0]}'/'sources'/station


def cached_observations(path, station, url, start, stop):
    """Approved instantaneous discharge for one cached page, clipped to its requested interval.

    The API interval includes both endpoints, so the boundary reading would repeat in the next
    quarter's page; clipping at the interval's end keeps every timestamp unique.
    """
    certificate = json.loads(path.with_suffix(path.suffix + '.request.json').read_text())
    if certificate['url'] != url:
        raise ValueError('Cached page does not match its declared interval: '+str(path))
    if digest(path) != certificate['sha256']:
        raise ValueError('Cached page checksum changed: '+str(path))
    payload = json.loads(gzip.decompress(path.read_bytes()))
    if payload.get('type') != 'FeatureCollection':
        raise ValueError('Expected USGS FeatureCollection: '+str(path))
    if [x for x in payload.get('links', []) if x.get('rel') == 'next']:
        raise ValueError('Unexpected pagination in a cached single-page response: '+str(path))
    series = parse_observations(payload['features'], station, '00011')
    return series.loc[(series.index >= start) & (series.index < stop)]


def hourly_station(station):
    """Complete hourly discharge calendar for one gauge, assembled from every cached page."""
    directory = cohort_directory(station)
    pieces, sources = [], {}
    for year in YEARS:
        for quarter, (start, end) in enumerate(intervals(year)):
            stop = pd.Timestamp(end)
            requested = (stop - pd.Timedelta(seconds=1)).isoformat() if quarter == 3 else end
            url = API + 'continuous/items?' + urlencode(dict(f='json',
                monitoring_location_id='USGS-' + station, parameter_code='00060',
                datetime=start + '/' + requested, limit=50000, skipGeometry='true'))
            path = directory/f'{year}-{quarter}-0{"-bounded" if quarter == 3 else ""}.json.gz'
            if not path.exists():
                continue
            pieces.append(cached_observations(path, station, url, pd.Timestamp(start), stop))
            sources[str(path.relative_to(ROOT))] = digest(path)
    if not pieces:
        raise FileNotFoundError('No cached discharge pages for '+station)
    series = pd.concat(pieces).sort_index()
    if series.index.has_duplicates:
        raise ValueError('Duplicate observation timestamps after clipping: '+station)
    return hourly_flow(series), sources


def cached_weather(station):
    """ERA5 hourly weather for one gauge, with the same validity contract the signals sprint enforces."""
    path = SIGNALS/'weather'/(station + '.json.gz')
    certificate = json.loads(path.with_suffix(path.suffix + '.request.json').read_text())
    if digest(path) != certificate['sha256']:
        raise ValueError('Weather checksum mismatch: '+str(path))
    hourly = json.loads(gzip.decompress(path.read_bytes()))['hourly']
    frame = pd.DataFrame(hourly)
    frame.index = pd.to_datetime(frame.pop('time'), utc=True)
    if frame.index.has_duplicates or not frame.index.equals(pd.date_range(frame.index[0], frame.index[-1], freq='h')):
        raise ValueError('Weather must use a complete unique UTC hourly calendar')
    if frame.snow_depth.notna().any():
        raise ValueError('Snow depth became available; the canonical variable set must change explicitly')
    if not np.isfinite(frame[WEATHER_VARIABLES].to_numpy(float)).all() or (frame.precipitation < 0).any():
        raise ValueError('Invalid cached weather values: '+station)
    return frame[WEATHER_VARIABLES]


def station_series(station, times, reference):
    """One gauge's four-channel hourly block over every hour the experiment queries, plus its origin.

    Channel 0 is log1p(discharge/reference); the weather channels sit at the canonical six-hour
    receipt delay, so the value a decision row reads at hour h is the weather of hour h-6.
    """
    if reference <= 0 or not len(times):
        raise ValueError('Require a positive reference discharge and at least one decision time')
    flow, sources = hourly_station(station)
    weather = cached_weather(station)
    origin = times.min() - pd.Timedelta(hours=WINDOW - 1)
    grid = pd.date_range(origin, times.max(), freq='h', tz='UTC')
    block = np.empty((len(CHANNELS), len(grid)), dtype=np.float32)
    block[0] = np.log1p(flow.reindex(grid).to_numpy(float)/reference)
    block[1:] = weather.reindex(grid - pd.Timedelta(hours=WEATHER_DELAY)).to_numpy(float).T
    return origin, block, sources


def windows(block, origin, times):
    """(rows, channels, WINDOW) slices ending at each decision time inclusive; no future values."""
    positions = ((times - origin).dt.total_seconds()/3600).to_numpy(int) - (WINDOW - 1)
    if positions.min() < 0 or positions.max() + WINDOW > block.shape[1]:
        raise ValueError('Decision window falls outside the assembled hourly series')
    view = np.lib.stride_tricks.sliding_window_view(block, WINDOW, axis=1)
    return np.moveaxis(view[:, positions], 1, 0)


def load_series(directory, records, station, cache=None):
    """Load one gauge's stored block, optionally keeping it in a per-process cache."""
    if cache is not None and station in cache:
        return cache[station]
    record = records[station]
    path = Path(directory)/record['file']
    if digest(path) != record['sha256']:
        raise ValueError('Stored series changed: '+str(path))
    block = np.load(path)
    if block.shape != (len(CHANNELS), record['hours']):
        raise ValueError('Stored series has an unexpected shape: '+str(path))
    loaded = (pd.Timestamp(record['origin']), block)
    if cache is not None:
        cache[station] = loaded
    return loaded

"""Independent confirmation run for the frozen six-hour high-flow candidate.

Protocol: docs/HIGH_FLOW_CONFIRMATION_PROTOCOL.md, frozen before any outcome for this
cohort was acquired or inspected. Cohort: the HUC2-stratified first batch of the 179
mutually disjoint catchments in experiments/high-flow-signals-20261018/fresh-geography/
camels-screen.json (disjoint_selection, recorded order). Candidate and comparator are the
cycle-11 artifacts applied unchanged, thresholds carried from the artifacts. Stage A
(00/06/18 UTC decisions) only; no HRRR fields exist for the cohort.

Commands, in order:
    python -m scripts.confirm_high_flow freeze    # pins cohort, hashes, rules, gates
    python -m scripts.confirm_high_flow collect   # USGS discharge, ERA5 weather, rows
    python -m scripts.confirm_high_flow score     # predictions, metrics, sixteen gates

Nothing here modifies production code, the frozen artifacts, or any earlier experiment.
"""
import argparse
import concurrent.futures
import gzip
import json
import time
from pathlib import Path
from urllib.parse import urlencode

import joblib
import numpy as np
import pandas as pd

from scripts import high_flow_signals as signals
from scripts import mlp_geo as archive
from scripts.flood_sprint import write_json
from scripts.high_flow_research import hydrologic_features
from scripts.hydrology_research import ROOT, IneligibleGauge, digest
from scripts.mlp_hybrid import cached_page
from scripts.mlp_hybrid_worker import physical_score
from scripts.shift_hydrology import API, FEATURES, decision_rows, hourly_flow
from scripts.shift_hydrology_metrics import jointly_detected_lead_delta, metrics, paired_uncertainty

OUT = ROOT/'experiments/high-flow-confirmation-20261018'
SIGNALS = ROOT/'experiments/high-flow-signals-20261018'
SCREEN = SIGNALS/'fresh-geography/camels-screen.json'
SITES = ROOT/'experiments/sprint-20261017-hydrology/research/sites.json'
CLIM = SIGNALS/'sources/camels_clim.txt'
PROTOCOL = ROOT/'docs/HIGH_FLOW_CONFIRMATION_PROTOCOL.md'
CANDIDATE_PATH = SIGNALS/'cycle-11/winner_forecast.joblib'
COMPARATOR_PATH = SIGNALS/'cycle-11/linear_conditional.joblib'
BATCH_SIZE = 20
HOURS = [0, 6, 18]
HORIZON = 6
OUTCOME_START = '2025-01-01T00:00:00Z'
OUTCOME_END = '2026-09-30T23:59:59Z'
SERIES_START = '2024-12-01T00:00:00Z'
SERIES_STOP = '2026-10-02T00:00:00Z'
HOURLY_YEARS = [2024, 2025, 2026]
WEATHER_START, WEATHER_END = '2019-12-01', '2026-09-30'
PRE_2023_BEGIN = pd.Timestamp('2022-01-01T00:00:00+00:00')
PRE_2023_END = pd.Timestamp('2021-12-31T00:00:00+00:00')
POOLED_CEILING, REGION_CEILING = 1.5, 3.0
DRAWS = 500
HRRR_COLUMNS = ['hrrr_point12', 'hrrr_basin12', 'hrrr_max12',
                'hrrr_point_ratio', 'hrrr_basin_ratio', 'hrrr_max_ratio']


def plain(value):
    """Recursively convert numpy scalars and timestamps so write_json can serialize."""
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def confirm_selection():
    """HUC2-stratified order over the recorded disjoint selection; batch 1 is its prefix."""
    screen = json.loads(SCREEN.read_text())
    selection = screen['disjoint_selection']
    regions, groups = [], {}
    for record in selection:
        region = record['huc'][:2]
        if region not in groups:
            groups[region] = []
            regions.append(region)
        groups[region].append(record)
    order, round_index = [], 0
    while len(order) < len(selection):
        added = False
        for region in regions:
            if round_index < len(groups[region]):
                order.append(groups[region][round_index])
                added = True
        if not added:
            break
        round_index += 1
    if len(order) != len(selection):
        raise ValueError('Stratified order lost stations')
    return screen, order, order[:BATCH_SIZE]


def station_records(batch):
    sites = {s['site_no']: s for s in json.loads(SITES.read_text())}
    cohort = []
    for record in batch:
        source = sites[record['station']]
        area, elevation = float(str(source['drain_area_va']).strip()), float(str(source['alt_va']).strip())
        cohort.append(dict(station=record['station'], name=record['name'], huc=record['huc'],
            site_huc=source['huc_cd'], latitude=float(source['dec_lat_va']),
            longitude=float(source['dec_long_va']), area_mi2=record['area_mi2'],
            area_km2=area*2.589988110336, elevation_m=elevation*.3048,
            role='confirmation'))
    return cohort


def p_mean_table():
    frame = pd.read_csv(CLIM, sep=';', dtype={'gauge_id': str})
    values = frame.set_index('gauge_id').p_mean
    return {key: float(value) for key, value in values.items()}


def artifact_summary(path, name):
    bundle = joblib.load(path)
    calibrator = bundle['calibrator']
    record = dict(name=name, path=str(path.relative_to(ROOT)), sha256=digest(path),
        threshold=float(bundle['threshold']), features=list(bundle['features']),
        physical='linear' if bundle['model'] is None else None,
        model=None if bundle['model'] is None else type(bundle['model']).__name__,
        calibrator_inputs=list(getattr(calibrator, 'feature_names_in_', [])))
    if bundle['model'] is None and name != 'linear_conditional':
        raise ValueError('Unexpected physical artifact layout')
    return record


def freeze():
    destination = OUT/'freeze.json'
    if destination.exists():
        raise FileExistsError(destination)
    screen, order, batch = confirm_selection()
    cohort = station_records(batch)
    clim = p_mean_table()
    missing = [s['station'] for s in cohort if s['station'] not in clim]
    if missing:
        raise ValueError('Missing CAMELS climatology: '+','.join(missing))
    if any(clim[s['station']] <= 0 for s in cohort):
        raise ValueError('Nonpositive precipitation climatology')
    candidate = artifact_summary(CANDIDATE_PATH, 'winner_forecast')
    comparator = artifact_summary(COMPARATOR_PATH, 'linear_conditional')
    if candidate['features'][-7:] != ['hrrr_available']+HRRR_COLUMNS:
        raise ValueError('Winner feature tail is not the expected HRRR block')
    document = dict(
        status='Frozen before any confirmation-cohort outcome was acquired or scored.',
        protocol=dict(path=str(PROTOCOL.relative_to(ROOT)), sha256=digest(PROTOCOL)),
        screen=dict(path=str(SCREEN.relative_to(ROOT)), sha256=digest(SCREEN)),
        sites=dict(path=str(SITES.relative_to(ROOT)), sha256=digest(SITES)),
        climatology=dict(path=str(CLIM.relative_to(ROOT)), sha256=digest(CLIM)),
        code_sha256=digest(Path(__file__)),
        artifacts=dict(candidate=candidate, comparator=comparator),
        provenance=dict(candidate_record=dict(path='experiments/high-flow-signals-20261018/candidate.json',
                sha256=digest(SIGNALS/'candidate.json')),
            cycle_freeze=dict(path='experiments/high-flow-signals-20261018/cycle-11/freeze.json',
                sha256=digest(SIGNALS/'cycle-11/freeze.json')),
            development_population='Retained 88-gauge pooled population; unchanged and never expanded with confirmation stations.'),
        cohort=dict(batch_size=BATCH_SIZE, n_selected=len(order),
            rule='HUC2 round-robin over the recorded order; batch 1 is the 20-station prefix',
            stratified_order=[record['station'] for record in order],
            batch=[record['station'] for record in batch], stations=cohort),
        acquisition=dict(
            reference='mlp_geo.reference: daily mean discharge 2000-2015, >=3000 accepted days, p95',
            hourly=dict(years=HOURLY_YEARS, reindex=[SERIES_START, SERIES_STOP],
                note='Quarter pages through scripts/mlp_geo.py observations; approved-only parser'),
            weather=dict(start=WEATHER_START, end=WEATHER_END, model='era5',
                variables=signals.VARIABLES, usable=signals.USABLE,
                note='Same request shape as high_flow_signals.acquire, extended end date'),
            series_span_probe='time-series-metadata instantaneous 00011 begin/end',
            hrrr='Absent for Stage A rows; hrrr_available=0, ratios and raw forecast fields NaN'),
        inclusion=dict(
            rules=['reference eligible (>=3000 pre-2016 daily records, positive p95)',
                   'instantaneous 00011 hourly series accepted by the approved-only parser',
                   'at least one eligible Stage A decision row'],
            exclusions=['missing_reference', 'missing_observations', 'boundary_problem',
                        'no_eligible_decisions'],
            no_events='A station with zero events in the window stays in the metrics as negative rows and is flagged in coverage.'),
        partitioning=dict(
            geographic='every included station (all Stage A rows)',
            temporal='stations whose instantaneous 00011 series begins before 2022-01-01 and ends on/after 2021-12-31 (pre-2023 records)',
            rule_notes=['The frozen candidate was fitted on the retained 88-gauge population only; no confirmation station contributes rows to the frozen fit.',
                        'A series-span probe is used instead of refitting anything: both artifacts, their calibrators and thresholds are applied unchanged.',
                        'Alternative reading of the protocol line about a refitted comparator (calibrator refit on cohort pre-2023 rows) is recorded as not executed; the task requires frozen artifacts.'],
            degeneracy='If the temporal row set equals the geographic row set, pooled statistics and total-event counts use the deduplicated union; disjoint partitions reproduce the standard concatenation.'),
        outcomes=dict(start=OUTCOME_START, end=OUTCOME_END, decision_hours=HOURS, horizon=HORIZON,
            stage='A', note='Stage B (12 UTC, HRRR) is not run and is never pooled with Stage A.'),
        features=dict(mirror='high_flow_signals.configurations(11) winner_forecast / population(11)',
            blocks=dict(features=FEATURES, weather6='rain_3..168_lag6, temperature_lag6, soil_lag6, positive_degree_24_lag6, rain_soil_lag6',
                ratios='rain24_ratio=rain_24_lag6/(2*p_mean); rain168_ratio=rain_168_lag6/(7*p_mean)',
                wetness='soil_change_6d=soil_lag6-soil_lag144; rain_share_6h=rain_6_lag6/(rain_168_lag6+1e-3)',
                hydro='high_flow_research.hydrologic_features', hrrr='absent for Stage A')),
        metrics=dict(functions=dict(
                metrics='scripts/shift_hydrology_metrics.py metrics',
                paired='scripts/shift_hydrology_metrics.py paired_uncertainty draws=500 seed=42 (HUC4 clusters)',
                lead='scripts/shift_hydrology_metrics.py jointly_detected_lead_delta',
                gate_evaluator='scripts/report_high_flow_signals.py promotion'),
            thresholds=dict(candidate=candidate['threshold'], comparator=comparator['threshold'])),
        gates={
            'at_least_20_events_per_partition': 'each partition has >= 20 events (checked by the two per-partition event gates; slot preserved for exact parity)',
            'at_least_50_events_total': 'total events >= 50 (checked by total_events; slot preserved for exact parity)',
            'temporal_test_events': 'temporal partition events >= 20',
            'temporal_test_ap': 'paired AP difference lower bound (500 HUC4-cluster draws) > 0',
            'temporal_test_event_gain': 'paired event-recall difference >= 0.05',
            'temporal_test_brier': 'candidate Brier <= comparator Brier',
            'temporal_test_false_alerts': 'pooled false alerts per 100 <= 1.5 and worst HUC2 <= 3.0',
            'temporal_test_lead': 'median lead delta on jointly detected events >= 0',
            'geographic_test_events': 'geographic partition events >= 20',
            'geographic_test_ap': 'paired AP difference lower bound (500 HUC4-cluster draws) > 0',
            'geographic_test_event_gain': 'paired event-recall difference >= 0.05',
            'geographic_test_brier': 'candidate Brier <= comparator Brier',
            'geographic_test_false_alerts': 'pooled false alerts per 100 <= 1.5 and worst HUC2 <= 3.0',
            'geographic_test_lead': 'median lead delta on jointly detected events >= 0',
            'total_events': 'events across partitions >= 50',
            'pooled_event_uncertainty': 'pooled paired event-recall lower bound (500 draws) > 0'})
    write_json(destination, plain(document))
    print('Frozen', len(cohort), 'stations; batch of', len(order), 'selected; candidate',
          candidate['sha256'][:12], 'comparator', comparator['sha256'][:12], flush=True)


def verify():
    frozen = json.loads((OUT/'freeze.json').read_text())
    for key in ['protocol', 'screen', 'sites', 'climatology']:
        if digest(ROOT/frozen[key]['path']) != frozen[key]['sha256']:
            raise ValueError('Frozen source changed: '+key)
    for record in frozen['artifacts'].values():
        if digest(ROOT/record['path']) != record['sha256']:
            raise ValueError('Frozen artifact changed: '+record['name'])
    if digest(Path(__file__)) != frozen['code_sha256']:
        raise ValueError('Implementation changed after freeze: '+str(Path(__file__)))
    return frozen


def fetch_weather(cohort, frozen):
    signals.OUT = OUT
    for station in cohort:
        path = OUT/'weather'/(station['station']+'.json.gz')
        if path.exists():
            continue
        params = dict(latitude=station['latitude'], longitude=station['longitude'],
            start_date=WEATHER_START, end_date=WEATHER_END, hourly=','.join(signals.VARIABLES),
            models='era5', timezone='GMT')
        payload = json.loads(signals.fetch(path, 'https://archive-api.open-meteo.com/v1/archive', params))
        if payload.get('error') or not all(v in payload.get('hourly', {}) for v in signals.VARIABLES):
            raise ValueError('Missing weather variables for '+station['station'])
        print('Weather', station['station'], len(payload['hourly']['time']), 'rows', flush=True)
        time.sleep(16)


def one(station, clim):
    sid = station['station']
    entry = dict(station=sid, name=station['name'], huc=station['huc'], site_huc=station['site_huc'])
    url = API + 'time-series-metadata/items?' + urlencode(dict(f='json',
        monitoring_location_id='USGS-'+sid, parameter_code='00060', limit=100))
    try:
        payload = cached_page(OUT/'sources'/sid/'series-metadata.json.gz', url)
        series = [f['properties'] for f in payload['features'] if f['properties'].get('statistic_id') == '00011']
        begins = [pd.Timestamp(s['begin']) for s in series if s.get('begin')]
        ends = [pd.Timestamp(s['end']) for s in series if s.get('end')]
        if not begins or not ends:
            raise ValueError('No instantaneous 00011 span in series metadata')
        entry['series_begin'] = min(begins).isoformat()
        entry['series_end'] = max(ends).isoformat()
        entry['pre_2023_records'] = bool(min(begins) < PRE_2023_BEGIN and max(ends) >= PRE_2023_END)
    except (ValueError, KeyError) as error:
        entry['pre_2023_records'] = None
        entry['metadata_note'] = str(error)
    try:
        reference = archive.reference(sid)
    except (IneligibleGauge, ValueError) as error:
        entry['excluded'] = 'missing_reference: '+str(error)
        return None, entry
    entry['reference_cfs'] = reference
    try:
        pieces = [archive.observations(sid, year) for year in HOURLY_YEARS]
        series = hourly_flow(pd.concat(pieces).sort_index())
        if series.empty:
            raise IneligibleGauge('No accepted instantaneous discharge')
        entry['accepted_hours_raw'] = int(series.notna().sum())
        q = series.reindex(pd.date_range(SERIES_START, SERIES_STOP, freq='h', tz='UTC', inclusive='left'))
        entry['accepted_hours'] = int(q.notna().sum())
        entry['total_hours'] = len(q)
    except (IneligibleGauge, ValueError) as error:
        entry['excluded'] = 'missing_observations: '+str(error)
        return None, entry
    try:
        rows, _, diagnostics = decision_rows(q, reference, station)
    except ValueError as error:
        entry['excluded'] = 'boundary_problem: '+str(error)
        return None, entry
    rows = rows.loc[rows.horizon.eq(HORIZON)].copy()
    times = pd.to_datetime(rows.decision_time, utc=True)
    keep = times.dt.hour.isin(HOURS) & times.ge(pd.Timestamp(OUTCOME_START)) & times.le(pd.Timestamp(OUTCOME_END))
    rows = rows.loc[keep].reset_index(drop=True)
    if rows.empty:
        entry['excluded'] = 'no_eligible_decisions'
        return None, entry
    try:
        path = OUT/'weather'/(sid+'.json.gz')
        hourly = json.loads(gzip.decompress(path.read_bytes()))['hourly']
        weather = signals.meteorology(hourly, 6)
        weather['soil_lag144'] = signals.meteorology(hourly, 144)['soil_lag144']
        aligned = weather.reindex(pd.to_datetime(rows.decision_time, utc=True))
        aligned.index = rows.index
        rows = pd.concat([rows, aligned], axis=1)
        lag6 = [column for column in rows if column.endswith('_lag6')] + ['soil_lag144']
        if rows[lag6].isna().any().any():
            raise ValueError('Incomplete weather history for eligible decisions')
        expectation = clim[sid]
        rows['p_mean'] = expectation
        rows['rain24_ratio'] = rows.rain_24_lag6/(2*expectation)
        rows['rain168_ratio'] = rows.rain_168_lag6/(7*expectation)
        rows['soil_change_6d'] = rows.soil_lag6 - rows.soil_lag144
        rows['rain_share_6h'] = rows.rain_6_lag6/(rows.rain_168_lag6 + 1e-3)
        rows = hydrologic_features(rows)
    except ValueError as error:
        entry['excluded'] = 'boundary_problem: '+str(error)
        return None, entry
    rows['hrrr_available'] = 0
    for column in HRRR_COLUMNS:
        rows[column] = np.nan
    events = rows.loc[rows.target.eq(1), 'event_time'].nunique()
    entry.update(included=True, decisions=int(len(rows)), positives=int(rows.target.sum()),
        events=int(events), onsets_total=int(len(diagnostics['events'])),
        first_decision=str(rows.decision_time.iloc[0]), last_decision=str(rows.decision_time.iloc[-1]))
    return rows, entry


def collect():
    frozen = verify()
    OUT.mkdir(parents=True, exist_ok=True)
    archive.OUT = OUT
    cohort = frozen['cohort']['stations']
    clim = p_mean_table()
    fetch_weather(cohort, frozen)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = {}
        futures = {pool.submit(one, station, clim): station['station'] for station in cohort}
        for future in concurrent.futures.as_completed(futures):
            station_id = futures[future]
            results[station_id] = future.result()
            entry = results[station_id][1]
            detail = entry.get('excluded') or 'included: %d decisions, %d events' % (entry['decisions'], entry['events'])
            print('Station', station_id, detail, flush=True)
    records = [results[station['station']] for station in cohort]
    frames = [frame for frame, _ in records if frame is not None and len(frame)]
    if not frames:
        raise ValueError('No usable confirmation data')
    rows = pd.concat(frames, ignore_index=True)
    if rows.duplicated(['station', 'decision_time', 'horizon']).any():
        raise ValueError('Duplicate station decisions')
    features = list(frozen['artifacts']['candidate']['features'])
    missing = [c for c in features if c not in rows]
    if missing:
        raise ValueError('Missing candidate features: '+','.join(missing))
    columns = ['station', 'huc', 'decision_time', 'horizon', 'target', 'event_time'] + features
    rows[columns].to_csv(OUT/'rows.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    coverage = [entry for _, entry in records]
    write_json(OUT/'coverage.json', plain(coverage))
    request_paths = sorted((OUT/'sources').rglob('*.request.json')) + sorted((OUT/'weather').glob('*.request.json'))
    requests = [json.loads(path.read_text()) for path in request_paths]
    included = [entry for entry in coverage if entry.get('included')]
    write_json(OUT/'acquisition.json', plain(dict(
        rows_sha256=digest(OUT/'rows.csv.gz'), rows=len(rows), stations=rows.station.nunique(),
        requests=len(requests),
        download_bytes=sum(int(r.get('bytes', r.get('uncompressed_bytes', 0))) for r in requests),
        request_files={str(path.relative_to(OUT)): json.loads(path.read_text())['sha256']
            for path in request_paths},
        included=len(included), excluded=len(coverage)-len(included),
        exclusions=[dict(station=e['station'], reason=e['excluded']) for e in coverage if e.get('excluded')],
        decisions=sum(int(e['decisions']) for e in included), events=sum(int(e['events']) for e in included),
        evidence='Retrospective approved USGS discharge and archived ERA5 reanalysis; the pipeline assumes a six-hour weather availability that is not as-issued receipt evidence.')))
    print('Collected', len(rows), 'Stage A rows across', rows.station.nunique(), 'stations;',
          len(coverage)-len(included), 'excluded', flush=True)


def apply_artifacts(frozen, frame):
    candidate, comparator = joblib.load(ROOT/frozen['artifacts']['candidate']['path']), \
        joblib.load(ROOT/frozen['artifacts']['comparator']['path'])
    for name, bundle, record in [('candidate', candidate, frozen['artifacts']['candidate']),
                                 ('comparator', comparator, frozen['artifacts']['comparator'])]:
        if int(round(float(bundle['threshold'])*1e12)) != int(round(record['threshold']*1e12)):
            raise ValueError('Threshold changed since freeze: '+name)
    probability = np.clip(candidate['model'].predict_proba(frame[list(candidate['features'])])[:, 1], 1e-6, 1-1e-6)
    score = np.log(probability/(1-probability))
    inputs = pd.DataFrame({'score': score})
    for column in ['season_sin', 'season_cos', 'margin', 'log_area']:
        inputs[column] = frame[column].to_numpy()
    p_candidate = candidate['calibrator'].predict_proba(inputs)[:, 1]
    inputs = pd.DataFrame({'score': physical_score('linear', frame)})
    for column in ['season_sin', 'season_cos', 'margin', 'log_area']:
        inputs[column] = frame[column].to_numpy()
    p_comparator = comparator['calibrator'].predict_proba(inputs)[:, 1]
    return p_candidate, p_comparator


def score():
    frozen = verify()
    frame = pd.read_csv(OUT/'rows.csv.gz', dtype={'station': str, 'huc': str}, keep_default_na=False)
    for column in HRRR_COLUMNS + ['hrrr_available']:
        frame[column] = pd.to_numeric(frame[column], errors='coerce')
    frame['hrrr_available'] = frame['hrrr_available'].fillna(0).astype(int)
    coverage = json.loads((OUT/'coverage.json').read_text())
    temporal = {entry['station'] for entry in coverage if entry.get('pre_2023_records') and entry.get('included')}
    thresholds = frozen['metrics']['thresholds']
    p_candidate, p_comparator = apply_artifacts(frozen, frame)
    frame = frame.reset_index(drop=True)
    masks = dict(temporal_validation=frame.station.isin(temporal).to_numpy(),
                 geographic_validation=np.ones(len(frame), dtype=bool))
    predictions = frame[['station', 'huc', 'decision_time', 'horizon', 'target', 'event_time']].copy()
    predictions['temporal_partition'] = masks['temporal_validation'].astype(int)
    predictions['probability_candidate'] = p_candidate
    predictions['probability_comparator'] = p_comparator
    predictions.to_csv(OUT/'predictions.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    results = dict(stage='A', outcome_window=[OUTCOME_START, OUTCOME_END], decision_hours=HOURS,
        horizon=HORIZON, rows=len(frame), stations=int(frame.station.nunique()),
        thresholds=thresholds, coverage=dict(
            included=sum(1 for e in coverage if e.get('included')),
            excluded=[dict(station=e['station'], reason=e['excluded']) for e in coverage if e.get('excluded')]),
        candidates={})
    for name, p, threshold in [('winner_forecast', p_candidate, thresholds['candidate']),
                               ('linear_conditional', p_comparator, thresholds['comparator'])]:
        entry = {}
        for partition, mask in masks.items():
            part = frame.loc[mask].reset_index(drop=True)
            measured = metrics(part, p[mask], threshold)
            regions = measured.pop('regions')
            entry[partition] = dict(measured, regions={key: {name: region[name] for name in
                ['rows', 'events', 'false_alerts_per_100', 'ap']} for key, region in regions.items()})
        results['candidates'][name] = dict(threshold=threshold, partitions=entry)
    write_json(OUT/'results.json', plain(results))
    entry, gates = promotion_record(frozen, frame, p_candidate, p_comparator, masks)
    def partition_summary(mask):
        part = frame.loc[mask]
        return dict(stations=int(part.station.nunique()), rows=int(mask.sum()),
                    events=int(part.loc[part.target.eq(1), 'event_time'].nunique()))
    record = dict(stage='A', partitions={name: partition_summary(mask) for name, mask in masks.items()},
        coincident=bool(masks['temporal_validation'].sum() == masks['geographic_validation'].sum()))
    record.update(entry)
    record['gates'] = gates
    record['passed'] = bool(all(gates.values()))
    record['candidate'] = frozen['artifacts']['candidate']['name']
    record['reference'] = frozen['artifacts']['comparator']['name']
    record['thresholds'] = thresholds
    record['evidence'] = ('Confirmation outcomes (2025-01-01..2026-09-30, Stage A) were acquired and inspected '
        'by this run; they can never again serve as untouched evidence.')
    write_json(OUT/'comparison.json', plain(record))
    for name in ['temporal_validation', 'geographic_validation']:
        values = entry[name]
        paired = values['paired']
        ap, gain, low = values['metrics']['ap'], paired['event_recall_difference'], paired['ap_difference_95']
        print(name, 'events', values['metrics']['events'],
              'candidate AP', None if ap is None else round(ap, 4),
              'recall gain', None if gain is None else round(gain, 4),
              'AP lower bound', None if not low else round(low[0], 4))
    print('gates', json.dumps(gates), 'passed', record['passed'], flush=True)


def promotion_record(frozen, frame, p_candidate, p_comparator, masks):
    """The sixteen gates exactly as scripts/report_high_flow_signals.py promotion()."""
    thresholds = frozen['metrics']['thresholds']
    entry = {}
    gates = {'at_least_20_events_per_partition': True, 'at_least_50_events_total': True}
    for partition, mask in masks.items():
        values = frame.loc[mask].reset_index(drop=True)
        pc, pr = p_candidate[mask], p_comparator[mask]
        measured = metrics(values, pc, thresholds['candidate'])
        regions = measured.pop('regions')
        warm = max((region['false_alerts_per_100'] for region in regions.values()), default=0.)
        delta = paired_uncertainty(values, pr, pc, thresholds['comparator'], thresholds['candidate'], draws=DRAWS)
        lead = jointly_detected_lead_delta(values, pr, pc, thresholds['comparator'], thresholds['candidate'])
        control = metrics(values, pr, thresholds['comparator'])
        control.pop('regions')
        brier_delta = float(measured['brier'] - control['brier'])
        entry[partition] = dict(metrics=measured,
            worst_region=max(regions, key=lambda key: regions[key]['false_alerts_per_100']) if regions else None,
            worst_region_false_alerts_per_100=float(warm),
            false_alert_ceiling_met=bool(measured['false_alerts_per_100'] <= POOLED_CEILING and warm <= REGION_CEILING),
            regions={key: dict(rows=value['rows'], events=value['events'],
                false_alerts_per_100=value['false_alerts_per_100'], ap=value['ap'])
                for key, value in regions.items()},
            paired=delta, lead=lead, brier_delta=brier_delta,
            control=dict(brier=control['brier'], ap=control['ap'], events=control['events'],
                detected_events=control['detected_events'], false_alerts_per_100=control['false_alerts_per_100']))
        prefix = partition.replace('_validation', '_test')
        low = delta.get('ap_difference_95')
        recall = delta.get('event_recall_difference')
        gates[prefix+'_events'] = bool(measured['events'] >= 20)
        gates[prefix+'_ap'] = bool(low and low[0] > 0)
        gates[prefix+'_event_gain'] = bool(recall is not None and recall >= .05)
        gates[prefix+'_brier'] = bool(brier_delta <= 0)
        gates[prefix+'_false_alerts'] = bool(entry[partition]['false_alert_ceiling_met'])
        gates[prefix+'_lead'] = bool(lead['median_lead_delta_hours'] is not None and lead['median_lead_delta_hours'] >= 0)
    unique = frame.loc[masks['temporal_validation'] | masks['geographic_validation']]
    unique = unique[~unique.duplicated(['station', 'decision_time', 'horizon'])]
    union_mask = frame.index.isin(unique.index)
    total = int(unique.loc[unique.target.eq(1)].event_time.nunique())
    pooled = paired_uncertainty(unique.reset_index(drop=True), p_comparator[union_mask], p_candidate[union_mask],
        thresholds['comparator'], thresholds['candidate'], draws=DRAWS)
    gates['total_events'] = bool(total >= 50)
    interval = pooled.get('event_recall_difference_95')
    gates['pooled_event_uncertainty'] = bool(interval and interval[0] > 0)
    entry['pooled'] = pooled
    entry['total_events_union'] = total
    coincident = bool(masks['temporal_validation'].sum() == masks['geographic_validation'].sum())
    entry['pooled_note'] = ('Partitions coincide; pooled statistics use the deduplicated union (the pre-registered '
        'degeneracy rule). The metrics validator rejects duplicate decisions, so a literal concatenation is not '
        'computable and would double-count the same evidence.') if coincident else \
        'Partitions are disjoint; pooled statistics equal the standard concatenation.'
    return entry, gates


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['freeze', 'collect', 'score'])
    arguments = parser.parse_args()
    {'freeze': freeze, 'collect': collect, 'score': score}[arguments.command]()

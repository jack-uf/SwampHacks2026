"""Sentinel-2 windowed COG reads: conventional indices and a Jev-ready true-colour render.

One scene window per (station, decision date): 5.12 km at 10 m, chosen by the lowest
local cloud fraction among recent acquisitions that were nominally available before the
decision time. Everything is cached under data/raw/s2/ keyed by the item identity.
"""
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import numpy as np
import rasterio
from rasterio.warp import transform as warp_transform
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'data/raw/s2'
STAC = 'https://earth-search.aws.element84.com/v1/search'
GDAL_ENV = dict(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', CPL_VSIL_CURL_ALLOWED_EXTENSIONS='.tif',
                GDAL_HTTP_TIMEOUT='30', GDAL_HTTP_MAX_RETRY='2', VSI_CACHE='TRUE')
HALF = 256  # 512 x 512 pixels at 10 m
LOOKBACK_DAYS, LATENCY_DAYS, MAX_SCENE_CLOUD, MAX_LOCAL_CLOUD, MAX_CANDIDATES = 21, 1, 60., .5, 4
BAD_SCL = [0, 1, 3, 8, 9, 10, 11]   # no-data, saturated, shadow, cloud, snow
CLOUD_SCL = [3, 8, 9, 10]


def candidates(lat, lon, decision_date):
    end = datetime.fromisoformat(decision_date) - timedelta(days=LATENCY_DAYS)
    start = end - timedelta(days=LOOKBACK_DAYS)
    body = dict(collections=['sentinel-2-l2a'], intersects=dict(type='Point', coordinates=[lon, lat]),
                datetime=f'{start.isoformat()}Z/{end.isoformat()}Z', limit=20,
                query={'eo:cloud_cover': {'lt': MAX_SCENE_CLOUD}})
    response = httpx.post(STAC, json=body, timeout=60)
    response.raise_for_status()
    items = sorted(response.json()['features'], key=lambda item: item['properties']['datetime'], reverse=True)
    return items[:MAX_CANDIDATES]


def read_window(href, lon, lat, size):
    with rasterio.Env(**GDAL_ENV):
        with rasterio.open('/vsicurl/' + href) as source:
            xs, ys = warp_transform('EPSG:4326', source.crs, [lon], [lat])
            row, col = source.index(xs[0], ys[0])
            window = Window(col - size // 2, row - size // 2, size, size)
            data = source.read(1, window=window, boundless=True, fill_value=0).astype('float32')
            return data, str(source.crs)


def scene(station, lat, lon, decision_date, half=HALF):
    CACHE.mkdir(parents=True, exist_ok=True)
    missing = dict(station=station, decision_date=decision_date, item=None, acquired=None, days_since=None,
                   missing=True, reason='no usable acquisition in the lookback window')
    items = candidates(lat, lon, decision_date)
    index_path = CACHE / '_local_cloud.json'
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    chosen = None
    for item in items:
        key = f"{station}_{item['id']}"
        if key in index:
            local_cloud = index[key]
            scl, crs = None, None
        else:
            scl, crs = read_window(item['assets']['scl']['href'], lon, lat, half)
            local_cloud = float(np.isin(scl, CLOUD_SCL).mean())
            index[key] = local_cloud
            index_path.write_text(json.dumps(index) + '\n')
        if chosen is None or local_cloud < chosen[1] - 1e-9:
            chosen = (item, local_cloud, scl, crs)
        if local_cloud == 0:
            break
    if chosen is None or chosen[1] > MAX_LOCAL_CLOUD:
        return missing
    item, local_cloud, scl, crs = chosen
    if scl is None:
        scl, crs = read_window(item['assets']['scl']['href'], lon, lat, half)
    png_path, json_path = CACHE / f'{station}_{item["id"]}.png', CACHE / f'{station}_{item["id"]}.json'
    if json_path.exists() and png_path.exists():
        return json.loads(json_path.read_text())
    started = time.time()
    red, _ = read_window(item['assets']['red']['href'], lon, lat, half * 2)
    green, _ = read_window(item['assets']['green']['href'], lon, lat, half * 2)
    blue, _ = read_window(item['assets']['blue']['href'], lon, lat, half * 2)
    nir, _ = read_window(item['assets']['nir']['href'], lon, lat, half * 2)
    red, green, blue, nir = red / 10000., green / 10000., blue / 10000., nir / 10000.
    invalid = (red <= 0) | (green <= 0) | (blue <= 0) | (nir <= 0)
    valid = ~invalid & ~np.kron(np.isin(scl, BAD_SCL), np.ones((2, 2), dtype=bool))
    ndvi = np.where(valid, (nir - red) / np.maximum(nir + red, 1e-6), np.nan)
    ndwi = np.where(valid, (green - nir) / np.maximum(green + nir, 1e-6), np.nan)
    scl_10m = np.kron(scl, np.ones((2, 2), dtype=bool))
    features = dict(station=station, decision_date=decision_date, item=item['id'],
                    acquired=item['properties']['datetime'], scene_cloud_cover=float(item['properties']['eo:cloud_cover']),
                    grid=item['properties'].get('grid:code'), crs=crs, window_px=half * 2, gsd_m=10, missing=False,
                    days_since=(datetime.fromisoformat(decision_date).replace(tzinfo=timezone.utc)
                                - datetime.fromisoformat(item['properties']['datetime'])).days,
                    local_cloud_fraction=local_cloud, valid_fraction=float(valid.mean()),
                    ndvi_mean=float(np.nanmean(ndvi)), ndvi_p10=float(np.nanpercentile(ndvi, 10)),
                    ndvi_p90=float(np.nanpercentile(ndvi, 90)),
                    ndwi_mean=float(np.nanmean(ndwi)), ndwi_p10=float(np.nanpercentile(ndwi, 10)),
                    ndwi_p90=float(np.nanpercentile(ndwi, 90)),
                    water_fraction=float(np.mean(np.isin(scl_10m[valid], [6]))),
                    vegetation_fraction=float(np.mean(np.isin(scl_10m[valid], [4]))),
                    bare_fraction=float(np.mean(np.isin(scl_10m[valid], [5]))),
                    read_seconds=round(time.time() - started, 1))
    rgb = np.stack([red, green, blue], axis=-1)
    low, high = np.nanpercentile(rgb[valid], 2), np.nanpercentile(rgb[valid], 98)
    image = np.clip((rgb - low) / max(high - low, 1e-6), 0, 1)
    image[~valid] = 0.92  # near-white marks masked pixels instead of hiding them
    from PIL import Image
    Image.fromarray((image * 255).astype('uint8')).save(png_path)
    json_path.write_text(json.dumps(features, indent=2) + '\n')
    return features


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('station')
    parser.add_argument('latitude', type=float)
    parser.add_argument('longitude', type=float)
    parser.add_argument('decision_date')
    args = parser.parse_args()
    print(json.dumps(scene(args.station, args.latitude, args.longitude, args.decision_date), indent=2))


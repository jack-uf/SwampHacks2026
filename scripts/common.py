"""Public-data download cache: retain bytes, URLs, retrieval times and checksums."""
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
import httpx
from email.utils import parsedate_to_datetime

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw'


def download(url: str, name: str, params=None) -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / name
    if path.exists() and path.with_suffix(path.suffix + '.source.json').exists():
        source = json.loads(path.with_suffix(path.suffix + '.source.json').read_text())
        if hashlib.sha256(path.read_bytes()).hexdigest() != source['sha256']:
            raise ValueError(f'Cached source checksum mismatch: {path}; remove this file and its manifest, then rerun ingestion')
        return path
    for attempt in range(4):
        try:
            response = httpx.get(url, params=params, timeout=120, follow_redirects=True,
                                 headers={'User-Agent': os.getenv('PLANIT_USER_AGENT', 'PlanITControl/0.1 research prototype')})
            response.raise_for_status()
            path.write_bytes(response.content)
            path.with_suffix(path.suffix + '.source.json').write_text(json.dumps({
                'url': str(response.url), 'retrieved_at': datetime.now(timezone.utc).isoformat(),
                'sha256': hashlib.sha256(response.content).hexdigest(), 'bytes': len(response.content)
            }, indent=2))
            return path
        except httpx.HTTPError as exc:
            if attempt == 3:
                raise
            delay = 2 ** attempt
            if isinstance(exc,httpx.HTTPStatusError) and exc.response.status_code == 429:
                if attempt > 0:
                    raise RuntimeError('Repeated provider rate limit; stop ingestion and retry after the provider window resets') from exc
                retry = exc.response.headers.get('Retry-After','60')
                try:
                    delay = float(retry)
                except ValueError:
                    delay = (parsedate_to_datetime(retry)-datetime.now(timezone.utc)).total_seconds()
                if delay > 60:
                    raise RuntimeError(f'Provider requests {retry} Retry-After; stop and rerun after that window') from exc
            time.sleep(max(0,delay))
    raise RuntimeError('Download did not complete')

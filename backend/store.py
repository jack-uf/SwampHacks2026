"""Single canonical SQLite schema for this local MVP. No historical-state migration."""
import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from scripts.common import ROOT


def db_path():
    return Path(os.getenv('PLANIT_DB', str(ROOT/'data/planit.sqlite')))


@contextmanager
def db():
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute('PRAGMA foreign_keys=ON')
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize():
    with db() as c:
        c.executescript('''
        CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, payload TEXT NOT NULL, fetched_at TEXT NOT NULL, error TEXT);
        CREATE TABLE IF NOT EXISTS hydrology_receipts (id TEXT PRIMARY KEY, station TEXT NOT NULL,
          received_at TEXT NOT NULL, url TEXT NOT NULL, status INTEGER, sha256 TEXT NOT NULL, body BLOB NOT NULL);
        CREATE INDEX IF NOT EXISTS hydrology_receipts_station_time ON hydrology_receipts(station, received_at);
        CREATE TABLE IF NOT EXISTS reports (id TEXT PRIMARY KEY, site_id TEXT NOT NULL, classification TEXT NOT NULL,
          description TEXT NOT NULL, latitude REAL NOT NULL, longitude REAL NOT NULL, created_at TEXT NOT NULL,
          status TEXT NOT NULL CHECK(status IN ('pending','verified','rejected','resolved')), image TEXT);
        CREATE TABLE IF NOT EXISTS report_triage (report_id TEXT PRIMARY KEY REFERENCES reports(id),
          payload TEXT NOT NULL, updated_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS strikes (id TEXT PRIMARY KEY, latitude REAL NOT NULL, longitude REAL NOT NULL,
          occurred_at TEXT NOT NULL, source TEXT NOT NULL CHECK(source IN ('live','replay')));
        CREATE TABLE IF NOT EXISTS briefings (id TEXT PRIMARY KEY, site_id TEXT NOT NULL, local_date TEXT NOT NULL,
          generated_at TEXT NOT NULL, payload TEXT NOT NULL, delivery TEXT NOT NULL DEFAULT 'not configured',
          UNIQUE(site_id, local_date));
        CREATE TABLE IF NOT EXISTS history (site_id TEXT NOT NULL, recorded_at TEXT NOT NULL, score REAL,
          coverage REAL NOT NULL, PRIMARY KEY(site_id, recorded_at));
        CREATE TABLE IF NOT EXISTS missions (id TEXT PRIMARY KEY, site_id TEXT NOT NULL, name TEXT NOT NULL, activity TEXT NOT NULL,
          latitude REAL NOT NULL, longitude REAL NOT NULL, shift_start TEXT NOT NULL, shift_end TEXT NOT NULL, crew_size INTEGER NOT NULL,
          supervisor TEXT NOT NULL, created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS jha_versions (mission_id TEXT NOT NULL REFERENCES missions(id), version INTEGER NOT NULL,
          created_at TEXT NOT NULL, trigger TEXT NOT NULL, fingerprint TEXT NOT NULL, payload TEXT NOT NULL, diff TEXT NOT NULL,
          PRIMARY KEY(mission_id, version));
        CREATE TABLE IF NOT EXISTS jha_decisions (id TEXT PRIMARY KEY, mission_id TEXT NOT NULL REFERENCES missions(id),
          version INTEGER NOT NULL, hazard TEXT NOT NULL, decision TEXT NOT NULL CHECK(decision IN ('accept','override','mitigate')),
          rating TEXT, mitigation TEXT, rationale TEXT NOT NULL, supervisor TEXT NOT NULL, entry_fingerprint TEXT NOT NULL, created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS jha_checks (mission_id TEXT PRIMARY KEY REFERENCES missions(id), checked_at TEXT NOT NULL,
          assessed_at TEXT NOT NULL, stale INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS window_decisions (id TEXT PRIMARY KEY, mission_id TEXT NOT NULL REFERENCES missions(id),
          created_at TEXT NOT NULL, actor TEXT NOT NULL, identity TEXT NOT NULL,
          choice TEXT NOT NULL CHECK(choice IN ('move','keep')), start TEXT, note TEXT,
          snapshot TEXT NOT NULL, jha_version INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS audit (seq INTEGER PRIMARY KEY AUTOINCREMENT, at TEXT NOT NULL, actor TEXT, identity TEXT NOT NULL,
          action TEXT NOT NULL, target TEXT NOT NULL, detail TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS jha_decision_identity (decision_id TEXT PRIMARY KEY REFERENCES jha_decisions(id),
          identity TEXT NOT NULL CHECK(identity IN ('token','self-declared')));
        ''')


def get_cache(key):
    with db() as c:
        row = c.execute('SELECT * FROM cache WHERE key=?', (key,)).fetchone()
    return dict(row, payload=json.loads(row['payload'])) if row else None


def put_cache(key, payload, fetched_at, error=None):
    with db() as c:
        c.execute('INSERT OR REPLACE INTO cache VALUES (?,?,?,?)', (key, json.dumps(payload), fetched_at, error))

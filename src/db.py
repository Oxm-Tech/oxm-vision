import sqlite3
import threading
import json
import os

_lock = threading.Lock()
_conn = None
_path = '/app/events.db'

def init(path='/app/events.db'):
    global _conn, _path
    _path = path
    _conn = sqlite3.connect(path, check_same_thread=False)
    _conn.execute("PRAGMA journal_mode=WAL")
    _conn.execute("PRAGMA synchronous=NORMAL")
    _conn.execute("PRAGMA cache_size=-8000")
    # Checkpoint automático cada 500 páginas (~2MB) para que el WAL
    # no crezca indefinidamente y los datos queden en el archivo principal.
    _conn.execute("PRAGMA wal_autocheckpoint=500")
    _conn.execute("""CREATE TABLE IF NOT EXISTS events (
        id    INTEGER PRIMARY KEY AUTOINCREMENT,
        ts    TEXT NOT NULL,
        type  TEXT NOT NULL,
        cam   TEXT,
        data  TEXT NOT NULL
    )""")
    _conn.execute("CREATE INDEX IF NOT EXISTS idx_ts      ON events(ts)")
    _conn.execute("CREATE INDEX IF NOT EXISTS idx_type    ON events(type)")
    _conn.execute("CREATE INDEX IF NOT EXISTS idx_ts_type ON events(ts, type)")
    _conn.commit()
    print(f"[DB] SQLite OK: {path}", flush=True)

def checkpoint():
    """Fuerza un WAL checkpoint completo. Llamar antes de cerrar/detener."""
    with _lock:
        try:
            result = _conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            print(f"[DB] Checkpoint: blocked={result[0]}, checkpointed={result[1]}, remaining={result[2]}", flush=True)
        except Exception as e:
            print(f"[DB] Checkpoint error: {e}", flush=True)

def write_event(record):
    with _lock:
        _conn.execute(
            "INSERT INTO events(ts, type, cam, data) VALUES(?,?,?,?)",
            (record.get('timestamp', ''), record.get('type', ''),
             record.get('cam'), json.dumps(record))
        )
        _conn.commit()

def write_events(records):
    if not records:
        return
    with _lock:
        _conn.executemany(
            "INSERT INTO events(ts, type, cam, data) VALUES(?,?,?,?)",
            [(r.get('timestamp', ''), r.get('type', ''), r.get('cam'), json.dumps(r))
             for r in records]
        )
        _conn.commit()

def tail_events(n=2000, types=None):
    """Returns last n events as dicts, oldest first."""
    with _lock:
        if types:
            ph = ','.join('?' * len(types))
            rows = _conn.execute(
                f"SELECT data FROM (SELECT data, id FROM events WHERE type IN ({ph}) "
                f"ORDER BY id DESC LIMIT ?) ORDER BY id ASC",
                [*types, n]
            ).fetchall()
        else:
            rows = _conn.execute(
                "SELECT data FROM (SELECT data, id FROM events ORDER BY id DESC LIMIT ?) "
                "ORDER BY id ASC",
                [n]
            ).fetchall()
    return [json.loads(r[0]) for r in rows]

def events_in_range(start_ts, end_ts=None, types=None):
    """Events from start_ts to end_ts (ISO strings), oldest first."""
    params = [start_ts]
    q = "SELECT data FROM events WHERE ts >= ?"
    if end_ts:
        q += " AND ts <= ?"
        params.append(end_ts)
    if types:
        ph = ','.join('?' * len(types))
        q += f" AND type IN ({ph})"
        params.extend(types)
    q += " ORDER BY ts ASC"
    with _lock:
        rows = _conn.execute(q, params).fetchall()
    return [json.loads(r[0]) for r in rows]

def time_range():
    """Returns (oldest_ts, newest_ts, total_count) or (None, None, 0) if empty."""
    with _lock:
        row = _conn.execute(
            "SELECT MIN(ts), MAX(ts), COUNT(*) FROM events"
        ).fetchone()
    return row if row else (None, None, 0)

def count_by_type():
    """Returns dict {type: count} for all event types."""
    with _lock:
        rows = _conn.execute(
            "SELECT type, COUNT(*) FROM events GROUP BY type ORDER BY COUNT(*) DESC"
        ).fetchall()
    return {r[0]: r[1] for r in rows}

def purge_detections(keep=50_000):
    """Conserva solo los últimos `keep` eventos de tipo detection."""
    try:
        with _lock:
            row = _conn.execute(
                "SELECT MIN(id) FROM (SELECT id FROM events WHERE type='detection' ORDER BY id DESC LIMIT ?)",
                [keep]
            ).fetchone()
            if row and row[0]:
                deleted = _conn.execute(
                    "DELETE FROM events WHERE type='detection' AND id < ?", [row[0]]
                ).rowcount
                _conn.commit()
                if deleted > 0:
                    print(f"[DB] Purge detections: {deleted} eliminados, conservando {keep}", flush=True)
    except Exception as e:
        print(f"[DB] Error purge_detections: {e}", flush=True)

def rotate_if_needed(max_mb=500):
    """Delete oldest 10% of rows if DB exceeds max_mb."""
    try:
        sz = os.path.getsize(_path) / 1024 ** 2
        if sz < max_mb:
            return
        with _lock:
            total = _conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            drop  = max(1, total // 10)
            _conn.execute(
                "DELETE FROM events WHERE id IN "
                "(SELECT id FROM events ORDER BY id ASC LIMIT ?)", [drop]
            )
            _conn.commit()
            _conn.execute("VACUUM")
        print(f"[DB] Rotación: {drop} eventos eliminados (DB era {sz:.0f}MB)", flush=True)
    except Exception as e:
        print(f"[DB] Error rotación: {e}", flush=True)

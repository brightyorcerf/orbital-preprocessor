"""
ground/scene_memory.py
──────────────────────
Episodic memory layer for the OSP ground station.

Persists anomaly detections across orbital passes in a local SQLite database.
Provides temporal and spatial queries to give the LLM analyst historical
context — turning each analysis from a stateless single-shot call into a
memory-augmented reasoning step.

Architecture role:
  engine.py → OSPPayload → scene_memory.remember()
                         ↓
  llm_analyst.py ← scene_memory.query_region() ← historical context
                         ↓
  dashboard.py ← scene_memory.get_timeline() ← trend display

Usage:
    from ground.scene_memory import SceneMemory
    mem = SceneMemory()
    mem.remember(payload)
    history = mem.query_region(lat=8.41, lon=77.82, radius_km=50)
"""

import json
import logging
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# Default DB path — sits next to this file in the ground/ directory
DEFAULT_DB_PATH = Path(__file__).parent / "osp_memory.db"


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class HistoricalAnomaly:
    """A single past detection record retrieved from memory."""
    scene_id:      str
    timestamp_utc: str
    anomaly_type:  str
    lat:           float
    lon:           float
    conf:          float
    alert_level:   Optional[str]   # ORION alert level assigned at the time
    pass_number:   int             # how many orbital passes ago (relative)

    def to_context_string(self) -> str:
        """
        Return a compact English description for LLM prompt injection.
        Keeps token count low while preserving all decision-relevant info.
        """
        age = f"{self.pass_number} pass(es) ago" if self.pass_number > 0 else "current pass"
        alert = f" [ORION: {self.alert_level}]" if self.alert_level else ""
        return (
            f"  • {self.anomaly_type.upper()} at ({self.lat:.4f}°, {self.lon:.4f}°), "
            f"conf={self.conf:.0%}, observed {age}{alert} [{self.timestamp_utc[:10]}]"
        )


@dataclass
class RegionHistory:
    """Summary of historical activity in a geographic region."""
    region_label:   str
    total_passes:   int
    anomaly_count:  int
    recurring_types: list[str]    # types seen more than once
    anomalies:      list[HistoricalAnomaly]
    first_seen:     Optional[str]
    last_seen:      Optional[str]

    @property
    def is_recurring(self) -> bool:
        return self.anomaly_count >= 2

    def to_context_string(self) -> str:
        """LLM-injectable summary of regional history."""
        if self.anomaly_count == 0:
            return f"No prior anomalies recorded in this region."

        lines = [
            f"Regional history ({self.region_label}): "
            f"{self.anomaly_count} anomaly detection(s) across {self.total_passes} orbital pass(es).",
        ]
        if self.recurring_types:
            lines.append(
                f"  Recurring object types: {', '.join(self.recurring_types).upper()}."
            )
        if self.first_seen and self.last_seen:
            lines.append(
                f"  Observed from {self.first_seen[:10]} to {self.last_seen[:10]}."
            )
        for a in self.anomalies[:5]:   # cap at 5 to control token usage
            lines.append(a.to_context_string())
        return "\n".join(lines)


# ── SceneMemory ────────────────────────────────────────────────────────────────

class SceneMemory:
    """
    SQLite-backed episodic memory for OSP detections.

    Thread-safe for single-process Streamlit use (SQLite WAL mode).
    Each orbital pass that is analysed is stored permanently; historical
    context can then be retrieved by region to augment LLM prompts.
    """

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self._conn   = self._connect()
        self._create_schema()
        log.info(f"SceneMemory initialised: {self.db_path}")

    # ── Connection ─────────────────────────────────────────────────────────────

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,   # Streamlit runs in threads
        )
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        return conn

    def _create_schema(self) -> None:
        """Create tables if they don't exist. Idempotent."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS scenes (
                scene_id        TEXT PRIMARY KEY,
                timestamp_utc   TEXT NOT NULL,
                lat_min         REAL,
                lat_max         REAL,
                lon_min         REAL,
                lon_max         REAL,
                cloud_cover     REAL,
                anomaly_count   INTEGER DEFAULT 0,
                alert_level     TEXT,          -- ORION output
                llm_summary     TEXT,          -- ORION summary text
                raw_payload_json TEXT          -- full JSON for audit
            );

            CREATE TABLE IF NOT EXISTS anomalies (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                scene_id        TEXT NOT NULL REFERENCES scenes(scene_id),
                timestamp_utc   TEXT NOT NULL,
                anomaly_type    TEXT NOT NULL,
                lat             REAL NOT NULL,
                lon             REAL NOT NULL,
                conf            REAL NOT NULL,
                bbox_px         TEXT           -- JSON array
            );

            CREATE INDEX IF NOT EXISTS idx_anomalies_latlon
                ON anomalies(lat, lon);
            CREATE INDEX IF NOT EXISTS idx_anomalies_type
                ON anomalies(anomaly_type);
            CREATE INDEX IF NOT EXISTS idx_scenes_timestamp
                ON scenes(timestamp_utc);
        """)
        self._conn.commit()

    # ── Write ──────────────────────────────────────────────────────────────────

    def remember(
        self,
        payload: dict,
        llm_brief: Optional[dict] = None,
    ) -> None:
        """
        Persist an OSPPayload (as dict/JSON) and optional ORION brief to memory.

        Args:
            payload:   OSPPayload as a Python dict (from json.loads(payload.to_json()))
            llm_brief: Optional ORION intelligence brief dict from llm_analyst
        """
        scene_id      = payload.get("scene_id", "UNKNOWN")
        timestamp_utc = payload.get("timestamp_utc", "")
        footprint     = payload.get("tile_footprint", {})
        anomalies     = payload.get("anomalies", [])
        alert_level   = llm_brief.get("alert_level")  if llm_brief else None
        llm_summary   = llm_brief.get("summary")      if llm_brief else None

        try:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO scenes
                  (scene_id, timestamp_utc, lat_min, lat_max, lon_min, lon_max,
                   cloud_cover, anomaly_count, alert_level, llm_summary, raw_payload_json)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    scene_id, timestamp_utc,
                    footprint.get("lat_min"), footprint.get("lat_max"),
                    footprint.get("lon_min"), footprint.get("lon_max"),
                    payload.get("cloud_cover", 0),
                    len(anomalies),
                    alert_level, llm_summary,
                    json.dumps(payload),
                ),
            )

            # Delete old anomaly rows if scene is being replaced
            self._conn.execute(
                "DELETE FROM anomalies WHERE scene_id = ?", (scene_id,)
            )

            for a in anomalies:
                ll = a.get("lat_lon", [0.0, 0.0])
                self._conn.execute(
                    """
                    INSERT INTO anomalies
                      (scene_id, timestamp_utc, anomaly_type, lat, lon, conf, bbox_px)
                    VALUES (?,?,?,?,?,?,?)
                    """,
                    (
                        scene_id, timestamp_utc,
                        a.get("type", "unknown"),
                        ll[0] if len(ll) > 0 else 0.0,
                        ll[1] if len(ll) > 1 else 0.0,
                        a.get("conf", 0.0),
                        json.dumps(a.get("bbox_px", [])),
                    ),
                )

            self._conn.commit()
            log.info(
                f"SceneMemory: remembered {scene_id} "
                f"({len(anomalies)} anomalies, alert={alert_level})"
            )

        except sqlite3.Error as e:
            log.error(f"SceneMemory write error: {e}")
            self._conn.rollback()

    # ── Read / Query ───────────────────────────────────────────────────────────

    @staticmethod
    def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Great-circle distance in kilometres."""
        R = 6371.0
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlam = math.radians(lon2 - lon1)
        a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
        return 2 * R * math.asin(math.sqrt(a))

    def query_region(
        self,
        lat: float,
        lon: float,
        radius_km: float = 50.0,
        limit: int = 20,
        exclude_scene_id: Optional[str] = None,
    ) -> RegionHistory:
        """
        Retrieve historical anomalies within radius_km of (lat, lon).

        Uses a bounding-box pre-filter (fast SQL) then Haversine refinement.
        Returns a RegionHistory object ready for LLM context injection.
        """
        # Approx degree-per-km for bounding box pre-filter
        dlat = radius_km / 111.0
        dlon = radius_km / (111.0 * math.cos(math.radians(lat)) + 1e-9)

        query = """
            SELECT a.scene_id, a.timestamp_utc, a.anomaly_type,
                   a.lat, a.lon, a.conf, s.alert_level
              FROM anomalies a
              LEFT JOIN scenes s ON a.scene_id = s.scene_id
             WHERE a.lat BETWEEN ? AND ?
               AND a.lon BETWEEN ? AND ?
        """
        params: list = [lat - dlat, lat + dlat, lon - dlon, lon + dlon]

        if exclude_scene_id:
            query += " AND a.scene_id != ?"
            params.append(exclude_scene_id)

        query += " ORDER BY a.timestamp_utc DESC LIMIT ?"
        params.append(limit * 3)  # over-fetch, then Haversine filter

        rows = self._conn.execute(query, params).fetchall()

        # Haversine refinement + pass numbering
        result_rows = []
        for i, r in enumerate(rows):
            dist = self._haversine_km(lat, lon, r["lat"], r["lon"])
            if dist <= radius_km:
                result_rows.append((i, r))

        result_rows = result_rows[:limit]

        anomalies = [
            HistoricalAnomaly(
                scene_id      = r["scene_id"],
                timestamp_utc = r["timestamp_utc"],
                anomaly_type  = r["anomaly_type"],
                lat           = r["lat"],
                lon           = r["lon"],
                conf          = r["conf"],
                alert_level   = r["alert_level"],
                pass_number   = rank,
            )
            for rank, (_, r) in enumerate(result_rows)
        ]

        # Compute summary stats
        type_counts: dict[str, int] = {}
        for a in anomalies:
            type_counts[a.anomaly_type] = type_counts.get(a.anomaly_type, 0) + 1
        recurring = [t for t, c in type_counts.items() if c >= 2]

        distinct_scenes = len({a.scene_id for a in anomalies})
        timestamps = sorted([a.timestamp_utc for a in anomalies])

        region_label = f"≈{radius_km:.0f}km around ({lat:.3f}°, {lon:.3f}°)"

        return RegionHistory(
            region_label   = region_label,
            total_passes   = distinct_scenes,
            anomaly_count  = len(anomalies),
            recurring_types = recurring,
            anomalies      = anomalies,
            first_seen     = timestamps[0]  if timestamps else None,
            last_seen      = timestamps[-1] if timestamps else None,
        )

    def get_timeline(
        self,
        limit: int = 50,
    ) -> list[dict]:
        """
        Return the most recent N scenes ordered by timestamp.
        Used by the dashboard timeline panel.
        """
        rows = self._conn.execute(
            """
            SELECT scene_id, timestamp_utc, anomaly_count,
                   cloud_cover, alert_level, llm_summary
              FROM scenes
             ORDER BY timestamp_utc DESC
             LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_anomaly_heatmap_data(self) -> list[dict]:
        """
        Return all stored anomaly locations for Folium heatmap layer.
        Each point: {lat, lon, weight (conf)}.
        """
        rows = self._conn.execute(
            "SELECT lat, lon, conf FROM anomalies ORDER BY conf DESC LIMIT 500"
        ).fetchall()
        return [{"lat": r["lat"], "lon": r["lon"], "weight": r["conf"]} for r in rows]

    def total_scenes(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM scenes").fetchone()[0]

    def total_anomalies(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM anomalies").fetchone()[0]

    def close(self) -> None:
        self._conn.close()
        log.info("SceneMemory: database connection closed.")


# ── Module-level singleton (convenience for dashboard) ─────────────────────────

_default_memory: Optional[SceneMemory] = None


def get_memory(db_path: str | Path = DEFAULT_DB_PATH) -> SceneMemory:
    """Return the module-level singleton SceneMemory, creating it if needed."""
    global _default_memory
    if _default_memory is None:
        _default_memory = SceneMemory(db_path)
    return _default_memory


# ── CLI demo ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import datetime
    import random

    mem = SceneMemory(db_path=":memory:")   # in-memory for demo

    # Seed with 3 fake historical passes
    for i in range(3):
        ts = (
            datetime.datetime.utcnow() - datetime.timedelta(hours=92 * (i + 1))
        ).strftime("%Y-%m-%dT%H:%M:%SZ")

        fake_payload = {
            "scene_id":      f"OSP-HIST-{i:04d}",
            "timestamp_utc": ts,
            "tile_footprint": {"lat_min": 8.0, "lat_max": 9.0,
                               "lon_min": 77.0, "lon_max": 78.0},
            "cloud_cover": round(random.uniform(0.05, 0.30), 2),
            "anomaly_count": 2,
            "anomalies": [
                {"type": "ship", "lat_lon": [8.41 + random.uniform(-0.05, 0.05),
                                             77.82 + random.uniform(-0.05, 0.05)],
                 "conf": round(random.uniform(0.70, 0.95), 2), "bbox_px": [300, 200, 360, 240]},
            ],
        }
        fake_brief = {"alert_level": ["GREEN", "YELLOW", "ORANGE"][i], "summary": f"Pass {i+1}"}
        mem.remember(fake_payload, fake_brief)

    # Query
    history = mem.query_region(lat=8.41, lon=77.82, radius_km=30)
    print(history.to_context_string())
    print(f"\nTotal scenes in memory: {mem.total_scenes()}")
    print(f"Total anomalies in memory: {mem.total_anomalies()}")

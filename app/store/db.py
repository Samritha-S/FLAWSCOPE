"""SQLite persistence store for speaker profiles and analysis sessions."""

from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional

DEFAULT_DB_PATH = Path("data/flawscope.db")


def get_connection(db_path: Optional[Path | str] = None) -> sqlite3.Connection:
    """Open SQLite connection with PRAGMA foreign_keys = ON and Row factory.

    Args:
        db_path: Path to SQLite database file. Defaults to DEFAULT_DB_PATH.

    Returns:
        sqlite3.Connection configured for foreign key constraints.
    """
    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Optional[Path | str] = None) -> Path:
    """Initialize database schema creating profiles and sessions tables.

    Args:
        db_path: Path to database. Defaults to data/flawscope.db.

    Returns:
        Path to initialized database file.
    """
    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    with get_connection(path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                goal TEXT,
                created_at TEXT NOT NULL,
                is_demo INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                profile_id INTEGER NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
                created_at TEXT NOT NULL,
                title TEXT,
                notes TEXT,
                baseline_label TEXT,
                participant_label TEXT,
                baseline_audio_path TEXT,
                participant_audio_path TEXT,
                transcript_text TEXT,
                transcript_sha1 TEXT,
                duration_p REAL,
                overall REAL,
                pace REAL,
                intonation REAL,
                volume REAL,
                pausing REAL,
                clarity REAL,
                n_flaws INTEGER,
                flaw_counts_json TEXT,
                result_json TEXT,
                is_demo INTEGER DEFAULT 0
            );
            """
        )
    return path


def create_profile(
    name: str,
    goal: str = "",
    is_demo: bool = False,
    db_path: Optional[Path | str] = None,
) -> int:
    """Create a new speaker profile.

    Args:
        name: Unique speaker profile name.
        goal: Target rhetorical/delivery goal.
        is_demo: Whether profile is a sample demo profile.
        db_path: Path to database.

    Returns:
        Newly created profile ID.
    """
    init_db(db_path)
    created_at = datetime.now(timezone.utc).isoformat()
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO profiles (name, goal, created_at, is_demo)
            VALUES (?, ?, ?, ?);
            """,
            (name, goal, created_at, int(is_demo)),
        )
        return int(cursor.lastrowid)


def list_profiles(db_path: Optional[Path | str] = None) -> List[Dict[str, Any]]:
    """List all speaker profiles.

    Args:
        db_path: Path to database.

    Returns:
        List of profile dictionaries ordered by created_at ASC.
    """
    init_db(db_path)
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT id, name, goal, created_at, is_demo
            FROM profiles
            ORDER BY created_at ASC, id ASC;
            """
        ).fetchall()
        return [dict(row) for row in rows]


def delete_profile(profile_id: int, db_path: Optional[Path | str] = None) -> None:
    """Delete speaker profile and cascade-delete all associated sessions.

    Args:
        profile_id: ID of profile to delete.
        db_path: Path to database.
    """
    init_db(db_path)
    with get_connection(db_path) as conn:
        conn.execute(
            """
            DELETE FROM profiles WHERE id = ?;
            """,
            (profile_id,),
        )


def save_session(
    profile_id: int,
    result: Dict[str, Any],
    ctx: Optional[Dict[str, Any]] = None,
    title: str = "",
    notes: str = "",
    is_demo: bool = False,
    db_path: Optional[Path | str] = None,
) -> int:
    """Persist an analysis session for a profile.

    Args:
        profile_id: Speaker profile ID.
        result: AnalysisResult dictionary output from core.pipeline.analyze.
        ctx: Optional runtime context with audio paths, labels, transcript.
        title: Optional session title.
        notes: Optional user notes.
        is_demo: Whether session is marked as demo.
        db_path: Path to database.

    Returns:
        Newly created session ID.
    """
    init_db(db_path)
    ctx = ctx or {}
    meta = result.setdefault("meta", {})
    scores = result.get("scores", {})
    flaws = result.get("flaws", [])

    # Ensure mode is explicitly set on meta
    if "mode" not in meta or not meta["mode"]:
        meta["mode"] = ctx.get("mode") or ("reference" if ctx.get("baseline_audio_path") else "benchmark")

    created_at = datetime.now(timezone.utc).isoformat()
    baseline_label = ctx.get("baseline_label") or meta.get("baseline_file", "")
    participant_label = ctx.get("participant_label") or meta.get("participant_file", "")
    baseline_audio_path = str(ctx.get("baseline_audio_path") or "")
    participant_audio_path = str(ctx.get("participant_audio_path") or "")

    transcript_text = ctx.get("transcript_text", "") or meta.get("transcript_text", "")
    if not transcript_text and "words" in result:
        transcript_text = " ".join(w.get("text", "") for w in result["words"])
    meta["transcript_text"] = transcript_text
    transcript_sha1 = hashlib.sha1(transcript_text.encode("utf-8")).hexdigest() if transcript_text else ""

    duration_p = float(meta.get("duration_p", 0.0))
    overall = float(scores.get("overall", 0.0))
    pace = float(scores.get("pace", 0.0))
    intonation = float(scores.get("intonation", 0.0))
    volume = float(scores.get("volume", 0.0))
    pausing = float(scores.get("pausing", 0.0))
    clarity = float(scores.get("clarity", 0.0))

    n_flaws = len(flaws)
    flaw_type_counts = Counter(f.get("type") for f in flaws if f.get("type"))
    flaw_counts_json = json.dumps(dict(flaw_type_counts))
    result_json = json.dumps(result)

    with get_connection(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO sessions (
                profile_id, created_at, title, notes,
                baseline_label, participant_label,
                baseline_audio_path, participant_audio_path,
                transcript_text, transcript_sha1,
                duration_p, overall, pace, intonation, volume, pausing, clarity,
                n_flaws, flaw_counts_json, result_json, is_demo
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                profile_id,
                created_at,
                title or f"Session {created_at[:10]}",
                notes,
                baseline_label,
                participant_label,
                baseline_audio_path,
                participant_audio_path,
                transcript_text,
                transcript_sha1,
                duration_p,
                overall,
                pace,
                intonation,
                volume,
                pausing,
                clarity,
                n_flaws,
                flaw_counts_json,
                result_json,
                int(is_demo),
            ),
        )
        return int(cursor.lastrowid)


def list_sessions(
    profile_id: int,
    db_path: Optional[Path | str] = None,
) -> List[Dict[str, Any]]:
    """List all sessions for a profile ordered oldest first.

    Args:
        profile_id: Profile ID.
        db_path: Path to database.

    Returns:
        List of session dictionaries sorted by created_at ASC, id ASC.
    """
    init_db(db_path)
    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT * FROM sessions
            WHERE profile_id = ?
            ORDER BY created_at ASC, id ASC;
            """,
            (profile_id,),
        ).fetchall()

        sessions = []
        for row in rows:
            d = dict(row)
            try:
                d["flaw_counts"] = json.loads(d.get("flaw_counts_json") or "{}")
            except Exception:
                d["flaw_counts"] = {}
            sessions.append(d)
        return sessions


def get_session(
    session_id: int,
    db_path: Optional[Path | str] = None,
) -> Optional[Dict[str, Any]]:
    """Retrieve a single session by ID.

    Args:
        session_id: Session ID.
        db_path: Path to database.

    Returns:
        Session dictionary or None if not found.
    """
    init_db(db_path)
    with get_connection(db_path) as conn:
        row = conn.execute(
            """
            SELECT * FROM sessions WHERE id = ?;
            """,
            (session_id,),
        ).fetchone()

        if not row:
            return None

        d = dict(row)
        try:
            d["flaw_counts"] = json.loads(d.get("flaw_counts_json") or "{}")
        except Exception:
            d["flaw_counts"] = {}
        try:
            d["result"] = json.loads(d.get("result_json") or "{}")
        except Exception:
            d["result"] = {}
        return d


def update_session(
    session_id: int,
    title: Optional[str] = None,
    notes: Optional[str] = None,
    db_path: Optional[Path | str] = None,
) -> None:
    """Update title and/or notes for an existing session.

    Args:
        session_id: Session ID.
        title: Optional new title.
        notes: Optional new notes.
        db_path: Path to database.
    """
    init_db(db_path)
    updates = []
    params = []

    if title is not None:
        updates.append("title = ?")
        params.append(title)
    if notes is not None:
        updates.append("notes = ?")
        params.append(notes)

    if not updates:
        return

    params.append(session_id)
    query = f"UPDATE sessions SET {', '.join(updates)} WHERE id = ?;"

    with get_connection(db_path) as conn:
        conn.execute(query, tuple(params))


def delete_session(session_id: int, db_path: Optional[Path | str] = None) -> None:
    """Delete a single session by ID.

    Args:
        session_id: Session ID.
        db_path: Path to database.
    """
    init_db(db_path)
    with get_connection(db_path) as conn:
        conn.execute("DELETE FROM sessions WHERE id = ?;", (session_id,))


def export_sessions_csv(
    profile_id: int,
    db_path: Optional[Path | str] = None,
) -> str:
    """Export all sessions for a profile as a CSV formatted string.

    Args:
        profile_id: Profile ID.
        db_path: Path to database.

    Returns:
        CSV string of sessions.
    """
    init_db(db_path)
    sessions = list_sessions(profile_id, db_path=db_path)
    if not sessions:
        return ""

    fieldnames = [
        "id",
        "profile_id",
        "created_at",
        "title",
        "notes",
        "baseline_label",
        "participant_label",
        "duration_p",
        "overall",
        "pace",
        "intonation",
        "volume",
        "pausing",
        "clarity",
        "n_flaws",
    ]

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for s in sessions:
        writer.writerow(s)
    return output.getvalue()

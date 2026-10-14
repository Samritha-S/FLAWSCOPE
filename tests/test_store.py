"""Unit tests for local profiles, session persistence, audio storage, and insights."""

import json
from pathlib import Path
import pytest
import sqlite3

from app.store.audio_store import store_session_audio
from app.store.db import (
    create_profile,
    delete_profile,
    delete_session,
    export_sessions_csv,
    get_session,
    init_db,
    list_profiles,
    list_sessions,
    save_session,
    update_session,
)
from app.store.insights import generate_insights, generate_insights_text


@pytest.fixture
def test_db(tmp_path: Path) -> Path:
    """Fixture providing a temporary SQLite database path."""
    db_file = tmp_path / "flawscope_test.db"
    return init_db(db_file)


def _make_mock_result(
    overall: float = 85.0,
    pace: float = 80.0,
    intonation: float = 85.0,
    volume: float = 90.0,
    pausing: float = 85.0,
    clarity: float = 85.0,
    flaws: list[dict] | None = None,
) -> dict:
    """Construct mock AnalysisResult dict for storage testing."""
    return {
        "meta": {
            "baseline_file": "clip01.wav",
            "participant_file": "participant01.wav",
            "duration_p": 45.2,
        },
        "scores": {
            "overall": overall,
            "pace": pace,
            "intonation": intonation,
            "volume": volume,
            "pausing": pausing,
            "clarity": clarity,
        },
        "flaws": flaws or [
            {"id": "F1", "type": "MONOTONE", "start": 10.0, "end": 12.5, "severity": 0.4},
            {"id": "F2", "type": "TOO_FAST", "start": 20.0, "end": 22.0, "severity": 0.5},
        ],
        "words": [{"idx": 0, "text": "Hello"}, {"idx": 1, "text": "world"}],
    }


def test_profile_lifecycle(test_db: Path):
    """Verify profile creation, listing, unique constraint, and deletion."""
    pid1 = create_profile("Elena Rostova", goal="TED Talk Preparation", db_path=test_db)
    pid2 = create_profile("Marcus Vance", goal="Debate Speed Control", is_demo=True, db_path=test_db)

    profiles = list_profiles(db_path=test_db)
    assert len(profiles) == 2
    assert profiles[0]["id"] == pid1
    assert profiles[0]["name"] == "Elena Rostova"
    assert profiles[1]["is_demo"] == 1

    # Unique constraint test
    with pytest.raises(sqlite3.IntegrityError):
        create_profile("Elena Rostova", db_path=test_db)


def test_session_lifecycle(test_db: Path):
    """Verify session persistence, retrieval, updating, and CSV export."""
    pid = create_profile("Speaker One", goal="Pacing", db_path=test_db)
    res = _make_mock_result(overall=82.5, pace=78.0)
    ctx = {
        "baseline_label": "Reference Churchill",
        "participant_label": "Take 1",
        "transcript_text": "Hello world from test transcript.",
    }

    sid = save_session(pid, res, ctx=ctx, title="Initial Practice", notes="Felt rushed", db_path=test_db)
    assert sid > 0

    session = get_session(sid, db_path=test_db)
    assert session is not None
    assert session["title"] == "Initial Practice"
    assert session["overall"] == 82.5
    assert session["pace"] == 78.0
    assert session["n_flaws"] == 2
    assert session["flaw_counts"]["MONOTONE"] == 1
    assert session["flaw_counts"]["TOO_FAST"] == 1
    assert len(session["transcript_sha1"]) == 40

    # Update session notes and title
    update_session(sid, title="Initial Practice (Revised)", notes="Clearer cadence", db_path=test_db)
    updated = get_session(sid, db_path=test_db)
    assert updated["title"] == "Initial Practice (Revised)"
    assert updated["notes"] == "Clearer cadence"

    # Export CSV
    csv_text = export_sessions_csv(pid, db_path=test_db)
    assert "id,profile_id,created_at,title" in csv_text
    assert "Initial Practice (Revised)" in csv_text


def test_cascade_delete(test_db: Path):
    """Verify deleting a profile cascades to delete all associated sessions."""
    pid = create_profile("Temporary Profile", db_path=test_db)
    save_session(pid, _make_mock_result(overall=80.0), db_path=test_db)
    save_session(pid, _make_mock_result(overall=85.0), db_path=test_db)

    assert len(list_sessions(pid, db_path=test_db)) == 2

    # Delete profile
    delete_profile(pid, db_path=test_db)

    # Sessions must be cascaded
    assert len(list_sessions(pid, db_path=test_db)) == 0


def test_audio_store_and_size_limit(tmp_path: Path, test_db: Path):
    """Verify audio copying, size limit check (>25 MB), and row update."""
    pid = create_profile("Audio Test Speaker", db_path=test_db)
    sid = save_session(pid, _make_mock_result(), db_path=test_db)

    audio_root = tmp_path / "audio_storage"

    # Create dummy 100 KB files
    b_file = tmp_path / "baseline_sample.wav"
    p_file = tmp_path / "part_sample.wav"
    b_file.write_bytes(b"\x00" * 102400)
    p_file.write_bytes(b"\x00" * 102400)

    stored_b, stored_p = store_session_audio(
        session_id=sid,
        profile_id=pid,
        baseline_audio_path=b_file,
        participant_audio_path=p_file,
        audio_root=audio_root,
        db_path=test_db,
    )

    assert stored_b is not None and Path(stored_b).exists()
    assert stored_p is not None and Path(stored_p).exists()
    assert Path(stored_b).name == f"{sid}_b.wav"
    assert Path(stored_p).name == f"{sid}_p.wav"

    # Verify session row updated
    s_row = get_session(sid, db_path=test_db)
    assert s_row["baseline_audio_path"] == stored_b
    assert s_row["participant_audio_path"] == stored_p

    # Test size limit: file over 25 MB skipped
    large_file = tmp_path / "huge_file.wav"
    with open(large_file, "wb") as f:
        f.seek((25 * 1024 * 1024) + 1024)
        f.write(b"\x00")

    stored_large, _ = store_session_audio(
        session_id=sid,
        profile_id=pid,
        baseline_audio_path=large_file,
        audio_root=audio_root,
        db_path=test_db,
    )
    assert stored_large is None


def test_insights_deterministic_templates():
    """Verify insights output for 1 session, 2 sessions, and 6 sessions."""
    # 1 Session
    s1 = [
        {
            "id": 1,
            "overall": 75.0,
            "pace": 70.0,
            "intonation": 72.0,
            "volume": 80.0,
            "pausing": 75.0,
            "clarity": 78.0,
            "flaw_counts": {"MONOTONE": 2, "TOO_FAST": 1},
        }
    ]
    txt1 = generate_insights_text(s1)
    assert txt1 == "Complete one more session to see your trend."
    data1 = generate_insights(s1)
    assert data1["ready"] is False

    # 2 Sessions
    s2 = list(s1) + [
        {
            "id": 2,
            "overall": 82.5,
            "pace": 80.0,  # +10.0
            "intonation": 76.0,  # +4.0
            "volume": 82.0,  # +2.0
            "pausing": 85.0,  # +10.0
            "clarity": 89.5,  # +11.5 (most improved)
            "flaw_counts": {"MONOTONE": 1, "UNCLEAR_ARTICULATION": 1},
        }
    ]
    data2 = generate_insights(s2)
    assert data2["ready"] is True
    assert "Overall score improved by +7.5 points" in data2["overall_sentence"]
    assert "Most improved dimension: Clarity (+11.5 points" in data2["improved_sentence"]
    assert "Weakest dimension: Intonation (76.0 points" in data2["weakest_sentence"]
    # MONOTONE appeared in session 1 and session 2 (2 of 2 sessions)
    assert "Most frequent flaw: Monotone (appeared in 2 of 2 sessions)" in data2["most_frequent_flaw"]

    # 6 Sessions
    s6 = list(s2) + [
        {"id": 3, "overall": 80.0, "pace": 78.0, "intonation": 75.0, "volume": 84.0, "pausing": 82.0, "clarity": 81.0, "flaw_counts": {"MONOTONE": 2}},
        {"id": 4, "overall": 86.0, "pace": 84.0, "intonation": 82.0, "volume": 88.0, "pausing": 88.0, "clarity": 88.0, "flaw_counts": {"TOO_SLOW": 1}},
        {"id": 5, "overall": 88.5, "pace": 86.0, "intonation": 85.0, "volume": 90.0, "pausing": 91.0, "clarity": 90.5, "flaw_counts": {"MONOTONE": 1}},
        {"id": 6, "overall": 92.0, "pace": 91.0, "intonation": 88.0, "volume": 94.0, "pausing": 95.0, "clarity": 92.0, "flaw_counts": {"LOW_VOLUME": 1}},
    ]
    data6 = generate_insights(s6)
    assert data6["ready"] is True
    assert "Overall score improved by +17.0 points (from 75.0 to 92.0)" in data6["overall_sentence"]
    # pace: 70->91 (+21), pausing: 75->95 (+20), intonation: 72->88 (+16)
    assert "Most improved dimension: Pace (+21.0 points" in data6["improved_sentence"]
    assert "Weakest dimension: Intonation (88.0 points in your latest session)" in data6["weakest_sentence"]
    # MONOTONE appeared in sessions 1, 2, 3, 5 (4 of 6 sessions)
    assert "Most frequent flaw: Monotone (appeared in 4 of 6 sessions)" in data6["most_frequent_flaw"]

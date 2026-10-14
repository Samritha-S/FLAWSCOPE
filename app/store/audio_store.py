"""Audio file persistence for saved speaker sessions."""

import os
from pathlib import Path
import shutil
from typing import Optional, Tuple

from app.store.db import get_connection

MAX_AUDIO_BYTES = 25 * 1024 * 1024  # 25 MB
DEFAULT_AUDIO_ROOT = Path("data/audio")


def store_session_audio(
    session_id: int,
    profile_id: int,
    baseline_audio_path: Optional[str | Path] = None,
    participant_audio_path: Optional[str | Path] = None,
    audio_root: Optional[Path | str] = None,
    db_path: Optional[Path | str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """Copy session audio files into data/audio/<profile>/<session>_b.wav and _p.wav and update session row.

    Files larger than 25 MB are skipped to prevent excessive storage growth.

    Args:
        session_id: Target session ID.
        profile_id: Owner profile ID.
        baseline_audio_path: Source path to baseline reference audio.
        participant_audio_path: Source path to participant audio.
        audio_root: Root directory for audio storage. Defaults to data/audio.
        db_path: Database path for row update.

    Returns:
        Tuple of (stored_baseline_path, stored_participant_path).
    """
    root = Path(audio_root) if audio_root else DEFAULT_AUDIO_ROOT
    target_dir = root / str(profile_id)
    target_dir.mkdir(parents=True, exist_ok=True)

    stored_b: Optional[str] = None
    stored_p: Optional[str] = None

    # Process baseline audio
    if baseline_audio_path:
        src_b = Path(baseline_audio_path)
        if src_b.exists() and src_b.is_file():
            if src_b.stat().st_size <= MAX_AUDIO_BYTES:
                dest_b = target_dir / f"{session_id}_b.wav"
                shutil.copyfile(src_b, dest_b)
                stored_b = str(dest_b)

    # Process participant audio
    if participant_audio_path:
        src_p = Path(participant_audio_path)
        if src_p.exists() and src_p.is_file():
            if src_p.stat().st_size <= MAX_AUDIO_BYTES:
                dest_p = target_dir / f"{session_id}_p.wav"
                shutil.copyfile(src_p, dest_p)
                stored_p = str(dest_p)

    # Update session row in database
    with get_connection(db_path) as conn:
        conn.execute(
            """
            UPDATE sessions
            SET baseline_audio_path = COALESCE(?, baseline_audio_path),
                participant_audio_path = COALESCE(?, participant_audio_path)
            WHERE id = ?;
            """,
            (stored_b, stored_p, session_id),
        )

    return stored_b, stored_p

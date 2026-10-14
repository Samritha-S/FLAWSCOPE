"""Transcript to speech word alignment via ASR timestamps and sequence matching."""

import difflib
import json
import re
import sys
from pathlib import Path
from typing import Any
import numpy as np

import config
from core.audio_io import file_sha1, load_audio

# Module-level cache for Whisper model
_WHISPER_MODEL = None

PUNCT_CHARS = {".", ",", ";", ":", "?", "!"}


def get_whisper_model():
    """Retrieve or lazily initialize module-level Whisper model on CPU."""
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        from faster_whisper import WhisperModel

        _WHISPER_MODEL = WhisperModel(
            config.ASR_MODEL,
            device="cpu",
            compute_type="int8",
        )
    return _WHISPER_MODEL


def _normalize_word(raw_token: str) -> str:
    """Normalize a token: lowercase, strip punctuation except inner apostrophes."""
    # Standardize curly apostrophes to ASCII straight apostrophe
    s = raw_token.lower().replace("’", "'").replace("`", "'")
    chars = []
    for i, ch in enumerate(s):
        if ch.isalnum():
            chars.append(ch)
        elif ch == "'" and 0 < i < len(s) - 1 and s[i - 1].isalnum() and s[i + 1].isalnum():
            chars.append(ch)
    return "".join(chars)


def tokenize_transcript(text: str) -> list[dict[str, Any]]:
    """Tokenize raw transcript text into normalized word tokens.

    Splits hyphenated words into separate tokens. Normalizes words to lowercase
    with punctuation removed except inner apostrophes. Sets has_punct_after to True
    when the raw token ends with . , ; : ? !.

    Args:
        text: Raw transcript string.

    Returns:
        List of dicts with keys: 'text', 'norm', 'has_punct_after'.
    """
    if not text:
        return []

    tokens: list[dict[str, Any]] = []
    # Split on whitespace first
    raw_words = text.split()

    for item in raw_words:
        # Standardize dashes and split hyphenated words
        item_dashes = item.replace("—", "-").replace("–", "-")
        sub_items = item_dashes.split("-")

        for idx_sub, sub in enumerate(sub_items):
            if not sub:
                continue

            # has_punct_after is True when raw token ends with . , ; : ? !
            # Check trailing character, or preceding character if wrapped in closing quotes
            has_punct = False
            stripped = sub.rstrip()
            if stripped:
                if stripped[-1] in PUNCT_CHARS:
                    has_punct = True
                elif len(stripped) > 1 and stripped[-1] in "\"'”’)" and stripped[-2] in PUNCT_CHARS:
                    has_punct = True

            norm = _normalize_word(sub)
            if norm:
                tokens.append({
                    "text": sub,
                    "norm": norm,
                    "has_punct_after": has_punct,
                })

    return tokens


def transcribe_words(y: np.ndarray) -> list[dict[str, Any]]:
    """Transcribe audio using faster-whisper and extract word-level timestamps.

    Args:
        y: Audio signal array (float32 at config.SR).

    Returns:
        List of dicts: [{'word': str, 'norm': str, 'start': float, 'end': float}].
    """
    model = get_whisper_model()
    segments, _ = model.transcribe(
        y,
        word_timestamps=True,
        beam_size=5,
        temperature=0.0,
        condition_on_previous_text=False,
        vad_filter=False,
        language="en",
    )

    asr_words: list[dict[str, Any]] = []
    for segment in segments:
        if segment.words:
            for w in segment.words:
                raw_w = w.word.strip()
                norm = _normalize_word(raw_w)
                if norm:
                    asr_words.append({
                        "word": raw_w,
                        "norm": norm,
                        "start": float(w.start),
                        "end": float(w.end),
                    })
    return asr_words


def align_transcript(
    tokens: list[dict[str, Any]],
    asr_words: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Align transcript tokens to ASR words using SequenceMatcher.

    Matches:
    - 'equal' blocks: copy start/end, matched=True.
    - 'replace' blocks of equal length: map 1:1, matched=True (spelling differences).
    - Unmatched transcript words: interpolate start/end linearly between nearest
      matched neighbours proportional to character length, matched=False.
    - ASR-only insertions/fillers are ignored.
    - Post-processes timestamps: monotonic, non-overlapping, min duration 0.04s, first start >= 0.

    Args:
        tokens: Output from tokenize_transcript.
        asr_words: Output from transcribe_words.

    Returns:
        List of AlignedWord dicts: idx, text, norm, has_punct_after, start, end, matched.
    """
    if not tokens:
        return []

    n_tokens = len(tokens)
    aligned: list[dict[str, Any]] = []
    for idx, token in enumerate(tokens):
        aligned.append({
            "idx": idx,
            "text": token["text"],
            "norm": token["norm"],
            "has_punct_after": token["has_punct_after"],
            "start": None,
            "end": None,
            "matched": False,
        })

    transcript_norms = [t["norm"] for t in tokens]
    asr_norms = [w["norm"] for w in asr_words]

    sm = difflib.SequenceMatcher(None, transcript_norms, asr_norms, autojunk=False)
    opcodes = sm.get_opcodes()

    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            for k in range(i2 - i1):
                aligned[i1 + k]["start"] = float(asr_words[j1 + k]["start"])
                aligned[i1 + k]["end"] = float(asr_words[j1 + k]["end"])
                aligned[i1 + k]["matched"] = True
        elif tag == "replace" and (i2 - i1) == (j2 - j1):
            # Equal length substitution: map 1:1 (handles spelling / transcription variances)
            for k in range(i2 - i1):
                aligned[i1 + k]["start"] = float(asr_words[j1 + k]["start"])
                aligned[i1 + k]["end"] = float(asr_words[j1 + k]["end"])
                aligned[i1 + k]["matched"] = True
        # For 'delete' or unequal 'replace', transcript words remain matched=False
        # 'insert' (ASR-only words / fillers) are ignored

    # Interpolate unmatched transcript words between nearest matched neighbours
    idx = 0
    while idx < n_tokens:
        if not aligned[idx]["matched"]:
            # Find contiguous block of unmatched words [start_gap, end_gap)
            start_gap = idx
            while idx < n_tokens and not aligned[idx]["matched"]:
                idx += 1
            end_gap = idx

            # Nearest left matched neighbor
            left_idx = start_gap - 1
            t_left = aligned[left_idx]["end"] if left_idx >= 0 else 0.0

            # Nearest right matched neighbor
            right_idx = end_gap
            t_right = aligned[right_idx]["start"] if right_idx < n_tokens else None

            # Calculate total character weights
            gap_words = aligned[start_gap:end_gap]
            char_counts = [max(1, len(w["norm"])) for w in gap_words]
            total_chars = sum(char_counts)

            if t_right is not None:
                # Interval bounded on right
                if t_right > t_left:
                    available_dur = t_right - t_left
                else:
                    available_dur = 0.04 * len(gap_words)

                cur_t = t_left
                for w, c in zip(gap_words, char_counts):
                    dur = available_dur * (c / total_chars)
                    w["start"] = cur_t
                    w["end"] = cur_t + dur
                    cur_t += dur
            else:
                # Unmatched at end of transcript
                cur_t = t_left
                for w, c in zip(gap_words, char_counts):
                    dur = max(0.04, c * 0.06)
                    w["start"] = cur_t
                    w["end"] = cur_t + dur
                    cur_t += dur
        else:
            idx += 1

    # Post-process: monotonic, non-overlapping times, minimum word duration 0.04 s, first start >= 0
    if aligned:
        # First start >= 0
        if aligned[0]["start"] is None or aligned[0]["start"] < 0.0:
            aligned[0]["start"] = 0.0
        if aligned[0]["end"] is None or aligned[0]["end"] < aligned[0]["start"] + 0.04:
            aligned[0]["end"] = aligned[0]["start"] + 0.04

        for i in range(1, n_tokens):
            cur = aligned[i]
            prev = aligned[i - 1]

            if cur["start"] is None or cur["start"] < prev["end"]:
                cur["start"] = prev["end"]
            if cur["end"] is None or cur["end"] < cur["start"] + 0.04:
                cur["end"] = cur["start"] + 0.04

        # Round timestamps to 3 decimal places for deterministic precision
        for w in aligned:
            w["start"] = round(float(w["start"]), 3)
            w["end"] = round(float(w["end"]), 3)

    return aligned


def align(
    y: np.ndarray,
    transcript_text: str,
    sha1: str | None = None,
) -> tuple[list[dict[str, Any]], float, list[str]]:
    """Align audio array to transcript text, optionally caching raw ASR output.

    Args:
        y: Audio signal array (float32 at config.SR).
        transcript_text: Reference transcript text.
        sha1: Optional audio file SHA1 hash for caching ASR output.

    Returns:
        Tuple of (words, matched_ratio, warnings).
    """
    asr_words = None
    cache_path = None

    if sha1:
        cache_dir = Path(".cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"asr_{sha1}.json"
        if cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    asr_words = json.load(f)
            except Exception:
                asr_words = None

    if asr_words is None:
        asr_words = transcribe_words(y)
        if cache_path:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(asr_words, f, indent=2)

    tokens = tokenize_transcript(transcript_text)
    words = align_transcript(tokens, asr_words)

    matched_count = sum(1 for w in words if w["matched"])
    matched_ratio = (matched_count / len(words)) if words else 0.0
    matched_ratio = round(matched_ratio, 4)

    warnings: list[str] = []
    if matched_ratio < 0.7:
        warnings.append(
            f"Low alignment match ratio: {matched_ratio:.1%} (< 70%). Check audio clarity or transcript accuracy."
        )

    return words, matched_ratio, warnings


def main():
    """CLI entrypoint: python -m core.alignment <audio_path> <transcript_path>."""
    if len(sys.argv) < 3:
        print("Usage: python -m core.alignment <audio_path> <transcript_path>")
        sys.exit(1)

    audio_path = Path(sys.argv[1])
    transcript_path = Path(sys.argv[2])

    if not audio_path.exists():
        print(f"Error: audio file not found: {audio_path}")
        sys.exit(1)
    if not transcript_path.exists():
        print(f"Error: transcript file not found: {transcript_path}")
        sys.exit(1)

    y, duration_s = load_audio(audio_path)
    sha1 = file_sha1(audio_path)

    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript_text = f.read()

    words, matched_ratio, warnings = align(y, transcript_text, sha1=sha1)

    # Print table
    print(f"\nAlignment Results for: {audio_path.name}")
    print(f"Duration: {duration_s:.2f}s | SHA1: {sha1[:8]}... | Matched Ratio: {matched_ratio:.1%}\n")
    print(f"{'idx':<5} | {'word':<20} | {'start (s)':<10} | {'end (s)':<10} | {'matched':<7}")
    print("-" * 62)
    for w in words:
        print(
            f"{w['idx']:<5} | {w['text']:<20} | {w['start']:<10.3f} | {w['end']:<10.3f} | {str(w['matched']):<7}"
        )

    if warnings:
        print("\nWarnings:")
        for warn in warnings:
            print(f"- {warn}")


if __name__ == "__main__":
    main()

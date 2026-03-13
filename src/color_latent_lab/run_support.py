from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    text = "\n".join(json.dumps(row, sort_keys=True) for row in rows)
    if text:
        text += "\n"
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_prediction_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            rows.append(json.loads(stripped))
    return rows


def _append_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True))
        handle.write("\n")


def _hash_words(words: list[str]) -> str:
    digest = hashlib.sha256()
    for word in words:
        digest.update(word.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _checkpoint_state_path(output_dir: Path, name: str) -> Path:
    return output_dir / "checkpoints" / f"{name}_state.json"


def _ensure_checkpoint_state(
    *,
    output_dir: Path,
    name: str,
    config: dict[str, Any],
    resume: bool,
) -> dict[str, Any]:
    state_path = _checkpoint_state_path(output_dir, name)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    if resume and state_path.exists():
        state = _read_json(state_path)
        if state.get("config") != config:
            raise ValueError(f"Checkpoint config mismatch for {name}; start with a fresh output dir.")
        return state
    state = {
        "completed_formats": [],
        "completed_patch_batches": [],
        "config": config,
        "selected_layers": None,
        "updated_at_utc": _utc_now(),
    }
    _write_json(state_path, state)
    return state


def _save_checkpoint_state(output_dir: Path, name: str, state: dict[str, Any]) -> None:
    state["updated_at_utc"] = _utc_now()
    _write_json(_checkpoint_state_path(output_dir, name), state)


class HeartbeatRecorder:
    def __init__(self, output_dir: Path, *, label: str) -> None:
        self.output_dir = output_dir
        self.label = label
        self.status_path = output_dir / "heartbeat_status.json"
        self.events_path = output_dir / "heartbeat_events.jsonl"
        self.start_time = time.monotonic()

    def write_manifest(self, **payload: Any) -> None:
        manifest = {
            "created_at_utc": _utc_now(),
            "cwd": str(Path.cwd()),
            "hostname": socket.gethostname(),
            "label": self.label,
            "pid": os.getpid(),
            "python_version": platform.python_version(),
        }
        manifest.update(payload)
        (self.output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def update(self, *, phase: str, message: str, **payload: Any) -> None:
        row = {
            "event": "heartbeat",
            "label": self.label,
            "message": message,
            "phase": phase,
            "runtime_seconds": round(time.monotonic() - self.start_time, 3),
            "state": payload.get("state", "running"),
            "updated_at_utc": _utc_now(),
        }
        row.update(payload)
        self.status_path.write_text(json.dumps(row, indent=2, sort_keys=True), encoding="utf-8")
        _append_jsonl_row(self.events_path, row)
        print(f"[{self.label}:{phase}] {message}", flush=True)

    def event(self, *, phase: str, message: str, **payload: Any) -> None:
        row = {
            "event": "info",
            "label": self.label,
            "message": message,
            "phase": phase,
            "runtime_seconds": round(time.monotonic() - self.start_time, 3),
            "updated_at_utc": _utc_now(),
        }
        row.update(payload)
        _append_jsonl_row(self.events_path, row)

    def fail(self, *, phase: str, error: BaseException) -> None:
        self.update(
            phase=phase,
            message=f"{type(error).__name__}: {error}",
            error_type=type(error).__name__,
            state="failed",
        )


__all__ = [
    "HeartbeatRecorder",
    "_append_jsonl_row",
    "_ensure_checkpoint_state",
    "_hash_words",
    "_read_json",
    "_read_prediction_rows",
    "_save_checkpoint_state",
    "_utc_now",
    "_write_json",
    "_write_jsonl",
]

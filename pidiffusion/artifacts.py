"""Run identifiers, protected run directories, and minimal manifests."""

from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_AMBIGUOUS_NAMES = {"latest", "final", "new", "updated"}
_REQUIRED_MANIFEST_FIELDS = {
    "schema_version",
    "run_id",
    "timestamp_utc",
    "status",
    "git",
    "source_files",
    "dataset",
    "checkpoint",
    "protocol",
    "randomness",
    "environment",
    "outputs",
}


def _validate_component(value: str, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    if value.lower() in _AMBIGUOUS_NAMES:
        raise ValueError(f"{name} cannot use ambiguous value: {value!r}")
    if not _SAFE_COMPONENT.fullmatch(value):
        raise ValueError(
            f"{name} contains unsafe characters: {value!r}; "
            "use letters, numbers, '.', '_' or '-'."
        )
    return value


def build_run_id(
    *,
    protocol: str,
    case_id: str,
    checkpoint_tag: str,
    seed: int,
    ddim_steps: int,
    timestamp_utc: datetime | None = None,
) -> str:
    """Build a UTC, filesystem-safe run identifier."""

    protocol = _validate_component(protocol, "protocol")
    case_id = _validate_component(case_id, "case_id")
    checkpoint_tag = _validate_component(checkpoint_tag, "checkpoint_tag")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if isinstance(ddim_steps, bool) or not isinstance(ddim_steps, int):
        raise ValueError("ddim_steps must be an integer")
    if ddim_steps <= 0:
        raise ValueError("ddim_steps must be positive")

    timestamp = timestamp_utc or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        raise ValueError("timestamp_utc must be timezone-aware")
    timestamp = timestamp.astimezone(timezone.utc)
    timestamp_text = timestamp.strftime("%Y%m%dT%H%M%SZ")
    return (
        f"{timestamp_text}_{protocol}_{case_id}_{checkpoint_tag}"
        f"_seed{seed}_ddim{ddim_steps}"
    )


def create_run_directory(
    results_root: str | Path,
    *,
    protocol: str,
    case_id: str,
    run_id: str,
) -> Path:
    """Create one run directory and refuse to reuse an existing path."""

    root = Path(results_root).expanduser().resolve()
    protocol = _validate_component(protocol, "protocol")
    case_id = _validate_component(case_id, "case_id")
    run_id = _validate_component(run_id, "run_id")
    target = root / protocol / case_id / run_id
    target.mkdir(parents=True, exist_ok=False)
    return target


def validate_manifest(manifest: dict[str, Any]) -> None:
    """Validate only the required top-level manifest fields."""

    if not isinstance(manifest, dict):
        raise TypeError("manifest must be a dictionary")
    missing = sorted(_REQUIRED_MANIFEST_FIELDS.difference(manifest))
    if missing:
        raise ValueError(
            "manifest is missing required top-level fields: "
            + ", ".join(missing)
        )


def write_manifest(run_dir: str | Path, manifest: dict[str, Any]) -> Path:
    """Atomically publish a new manifest without overwriting an existing one."""

    validate_manifest(manifest)
    directory = Path(run_dir).expanduser().resolve()
    if not directory.exists():
        raise FileNotFoundError(f"Run directory does not exist: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Run path is not a directory: {directory}")

    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Manifest already exists: {manifest_path}")

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=directory,
            prefix=".manifest.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(
                manifest,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        # A same-directory hard link publishes atomically and refuses overwrite.
        os.link(temporary_path, manifest_path)
        return manifest_path
    except FileExistsError:
        raise
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def update_manifest(run_dir: str | Path, manifest: dict[str, Any]) -> Path:
    """Atomically update an existing manifest without changing its identity."""

    validate_manifest(manifest)
    directory = Path(run_dir).expanduser().resolve()
    if not directory.exists():
        raise FileNotFoundError(f"Run directory does not exist: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"Run path is not a directory: {directory}")

    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        existing_manifest = json.load(handle)
    validate_manifest(existing_manifest)
    if manifest["schema_version"] != existing_manifest["schema_version"]:
        raise ValueError("Manifest schema_version cannot change")
    if manifest["run_id"] != existing_manifest["run_id"]:
        raise ValueError("Manifest run_id cannot change")
    if existing_manifest.get("status") == "completed":
        raise RuntimeError("Completed manifests cannot be updated")

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=directory,
            prefix=".manifest.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(
                manifest,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, manifest_path)
        return manifest_path
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

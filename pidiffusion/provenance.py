"""Small, explicit provenance helpers for PIDiffusion runs."""

from __future__ import annotations

import hashlib
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _as_path(path: str | Path) -> Path:
    return Path(path).expanduser()


def _utc_iso(timestamp: float) -> str:
    return (
        datetime.fromtimestamp(timestamp, tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the lowercase SHA-256 of one file using streaming reads."""

    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")

    file_path = _as_path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File does not exist: {file_path}")
    if not file_path.is_file():
        raise IsADirectoryError(f"Expected a regular file: {file_path}")

    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_git(repo_root: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("Git executable was not found") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout).strip()
        raise RuntimeError(
            f"Git command failed ({' '.join(arguments)}): {detail}"
        ) from exc
    return completed.stdout.strip()


def git_state(repo_root: str | Path) -> dict[str, Any]:
    """Read branch, commit, and porcelain status without changing Git state."""

    root = _as_path(repo_root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Repository root is not a directory: {root}")

    status_output = _run_git(root, "status", "--porcelain=v1")
    status_porcelain = status_output.splitlines() if status_output else []
    return {
        "branch": _run_git(root, "rev-parse", "--abbrev-ref", "HEAD"),
        "sha": _run_git(root, "rev-parse", "HEAD"),
        "dirty": bool(status_porcelain),
        "untracked": any(line.startswith("??") for line in status_porcelain),
        "status_porcelain": status_porcelain,
    }


def runtime_environment() -> dict[str, Any]:
    """Return lightweight runtime information without model or tensor setup."""

    executable_path = Path(sys.executable).resolve()
    conda_environment = os.environ.get("CONDA_DEFAULT_ENV")
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if (
        conda_prefix is None
        and executable_path.parent.name == "bin"
        and executable_path.parent.parent.parent.name == "envs"
    ):
        conda_prefix = str(executable_path.parent.parent)
        if conda_environment is None:
            conda_environment = executable_path.parent.parent.name

    environment: dict[str, Any] = {
        "python_version": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "conda_environment": conda_environment,
        "conda_prefix": conda_prefix,
        "pytorch_version": None,
        "cuda_build_version": None,
        "cuda_available": None,
        "cuda_device_count": None,
        "visible_gpu_names": None,
        "pytorch_import_error": None,
        "cuda_probe_error": None,
    }

    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment dependent
        environment["pytorch_import_error"] = f"{type(exc).__name__}: {exc}"
        return environment

    environment["pytorch_version"] = torch.__version__
    environment["cuda_build_version"] = torch.version.cuda
    try:
        available = bool(torch.cuda.is_available())
        environment["cuda_available"] = available
        count = int(torch.cuda.device_count()) if available else 0
        environment["cuda_device_count"] = count
        environment["visible_gpu_names"] = (
            [torch.cuda.get_device_name(index) for index in range(count)]
            if available
            else []
        )
    except Exception as exc:  # CUDA may be unavailable or misconfigured.
        environment["cuda_probe_error"] = f"{type(exc).__name__}: {exc}"
    return environment


def file_identity(path: str | Path) -> dict[str, Any]:
    """Describe one explicitly supplied file without scanning directories."""

    original = _as_path(path)
    resolved = original.resolve(strict=False)
    identity: dict[str, Any] = {
        "path": str(path),
        "resolved_path": str(resolved),
        "exists": original.exists(),
        "size_bytes": None,
        "mtime_utc": None,
        "sha256": None,
    }
    if not original.exists():
        return identity

    stat_result = original.stat()
    identity["size_bytes"] = stat_result.st_size
    identity["mtime_utc"] = _utc_iso(stat_result.st_mtime)
    if original.is_file():
        identity["sha256"] = sha256_file(original)
    return identity

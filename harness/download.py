"""Download and checksum helpers shared by the COCO harness and the dataset loaders."""

from __future__ import annotations

import hashlib
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path

USER_AGENT = "vtas-harness/1.0 (+https://github.com/Ritanshu-Prasad/VTAS_Evaluation_Metric)"


class ChecksumError(RuntimeError):
    pass


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def verify(path: str | Path, sha256: str | None) -> Path:
    """Raise :class:`ChecksumError` unless ``path`` has the expected SHA-256."""
    path = Path(path)
    if sha256 is None:
        return path
    actual = sha256_file(path)
    if actual != sha256:
        raise ChecksumError(f"{path}: sha256 {actual} != expected {sha256}")
    return path


def fetch(url: str, dest: str | Path, sha256: str | None = None, timeout: int = 120,
          retries: int = 3) -> Path:
    """Download ``url`` to ``dest`` (skipped when it exists and verifies), then verify.

    Network errors are retried; a checksum mismatch is not (it means the source changed).
    """
    dest = Path(dest)
    if dest.exists():
        try:
            return verify(dest, sha256)
        except ChecksumError:
            dest.unlink()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, length=1 << 20)
            break
        except OSError:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    verify(tmp, sha256)
    tmp.replace(dest)
    return dest


def extract_zip(archive: str | Path, members: list[str] | None, out_dir: str | Path) -> Path:
    """Extract ``members`` (all when None) of a zip, refusing paths outside ``out_dir``."""
    out_dir = Path(out_dir).resolve()
    with zipfile.ZipFile(archive) as z:
        for name in members or z.namelist():
            target = (out_dir / name).resolve()
            if out_dir not in target.parents and target != out_dir:
                raise ValueError(f"unsafe path in archive: {name}")
            if not target.exists():
                z.extract(name, out_dir)
    return out_dir

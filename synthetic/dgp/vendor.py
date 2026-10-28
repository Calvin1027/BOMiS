"""C21: 박사형 코드 고정 사본(fa088a9) 불러오기와 해시 확인 (설계 6.1)."""

import hashlib
import importlib
import sys
from pathlib import Path

VENDOR_ROOT = Path(__file__).resolve().parents[1] / "vendor" / "bomis_fa088a9"


def load_bomis():
    # C21, D21.1: sys.path 앞에 사본 루트를 넣고 import한다.
    if str(VENDOR_ROOT) not in sys.path:
        sys.path.insert(0, str(VENDOR_ROOT))
    pkg = importlib.import_module("bomis")
    importlib.import_module("bomis.model")
    importlib.import_module("bomis.metrics")
    return pkg


def verify_hashes() -> dict[str, bool]:
    # C21, D21.1: SHA256SUMS의 각 줄을 다시 계산해 비교한다.
    out = {}
    for line in (VENDOR_ROOT / "SHA256SUMS").read_text().splitlines():
        digest, rel = line.split(maxsplit=1)
        rel = rel.strip()
        out[rel] = hashlib.sha256((VENDOR_ROOT / rel).read_bytes()).hexdigest() == digest
    return out

from __future__ import annotations

import hashlib


def semantic_seed(root: int, *roles: object) -> int:
    payload = "\x1f".join([str(int(root)), *(str(role) for role in roles)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31 - 1)


def assert_distinct_seed_roles(*seeds: int) -> None:
    if len({int(seed) for seed in seeds}) != len(seeds):
        raise ValueError("semantic seed roles must be distinct")

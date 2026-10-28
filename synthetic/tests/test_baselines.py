import numpy as np

from baselines import external
from baselines.ros import ros


def test_ros():
    # V26
    D1 = np.random.default_rng(0).integers(0, 4, (50, 8))
    out = ros(D1, 200, np.random.default_rng(1))
    keys = {r.tobytes() for r in D1.astype(np.int64)}
    assert out.shape == (200, 8) and all(r.tobytes() in keys for r in out)


def test_train_table_format():
    D0 = np.zeros((3, 8), int)
    D1 = np.ones((2, 8), int)
    f = external.train_table(D0, D1)
    assert list(f.columns) == [f"c{j}" for j in range(8)] + ["target"]
    assert f["target"].tolist() == ["0", "0", "0", "1", "1"] and f["c0"].tolist()[-1] == "1"


def test_external_failure_recorded(tmp_path, monkeypatch):
    # Review Focus 3: 인터프리터가 실패해도 예외 없이 실패로 돌려준다(재시도 포함).
    monkeypatch.setenv("BL_PYTHON", "/bin/false")
    monkeypatch.setenv("BL_DEPS", str(tmp_path))
    cfg = tmp_path / "cfg.json"
    cfg.write_text('{"models": {"smoten": {"k_neighbors": 5}}}')
    D = np.random.default_rng(0).integers(0, 4, (20, 8))
    res = external.run_external("smoten", D, D[:5], 4, 0, tmp_path / "w", base_config=cfg)
    assert res["ok"] is False and res["rows"] is None and res["attempt"] == -1

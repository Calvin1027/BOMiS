from dgp.vendor import load_bomis, verify_hashes


def test_vendor_hashes():
    # V20
    res = verify_hashes()
    assert len(res) == 6 and all(res.values())


def test_vendor_threshold_values():
    # V19
    model = load_bomis().model
    got = [model.certify_threshold(8000, n1, 0.5, 0.05 / 480) for n1 in (50, 100, 200, 400, 800)]
    assert got == [2603, 1346, 680, 340, 170]

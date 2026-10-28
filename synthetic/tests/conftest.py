import pytest

from dgp.population import generate_accepted


@pytest.fixture(scope="session")
def pop():
    # 테스트용 모집단: seed 9000부터 처음 수락된 것.
    pops, _ = generate_accepted(9000, 1, max_tries=50)
    return pops[0]

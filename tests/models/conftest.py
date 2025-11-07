from unittest.mock import Mock

import pytest


@pytest.fixture
def mock_surprise_algorithm():
    mock_algo = Mock()
    mock_algo.fit.return_value = mock_algo
    mock_algo.predict.return_value = Mock(est=4.0)
    mock_algo.test.return_value = [Mock(est=4.0, r_ui=4.0)]
    return mock_algo


@pytest.fixture
def mock_trainset():
    return Mock()


@pytest.fixture
def mock_testset():
    return [("user1", "item1", 4.0)]

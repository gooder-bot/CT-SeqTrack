import pytest
from tests.helpers.frozen_source import prepare_frozen_source


@pytest.fixture(scope='session')
def comparison_root():
    return prepare_frozen_source()

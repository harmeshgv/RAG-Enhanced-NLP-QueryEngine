import pytest
from dev.core.preprocessing import PreProcessor

@pytest.fixture
def processor():
    return PreProcessor(param=None)

def test_to_lowercase(processor):
    assert processor.to_lowercase(["HELLO", "World"]) == ["hello", "world"]
    
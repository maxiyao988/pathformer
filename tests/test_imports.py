import pytest

try:
    from pathformer.models import Model
    from pathformer.data_provider.data_factory import data_provider
    from pathformer.utils.tools import adjust_learning_rate
except ImportError as e:
    pytest.skip(f"Skipping import tests because dependencies are missing: {e}")


def test_package_imports():
    assert Model is not None
    assert callable(data_provider)
    assert callable(adjust_learning_rate)

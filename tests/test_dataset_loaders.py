import pytest

from llamafactory.dataset_loaders import DatasetFactory, DatasetName


@pytest.mark.tufa
def test_dataset_factory_registry() -> None:
    assert set(DatasetFactory.registered_loaders) == set(DatasetName)

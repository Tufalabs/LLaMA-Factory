from abc import ABC, abstractmethod
from collections.abc import Generator, Iterable
from typing import Any

from datasets import Dataset

from .utils import extract_train_test


TEST_FRACTION_OF_TRAIN = 0.02


class BaseDatasetLoader(ABC):
    @classmethod
    @abstractmethod
    def get_full_subset_size(cls) -> int | float:
        pass

    @classmethod
    @abstractmethod
    def load(cls) -> Dataset:
        pass

    @classmethod
    @abstractmethod
    def format_dataset(cls, data: Iterable[dict[str, Any]]) -> Generator[dict[str, Any], None, None]:
        pass

    @classmethod
    def get_data_subset(cls, dataset: Dataset, seed: int) -> Dataset:
        subset_size = cls.get_full_subset_size()
        # train_size >= dataset_size raises ValueError
        if subset_size >= len(dataset):
            return dataset

        return dataset.train_test_split(
            train_size=subset_size,
            seed=seed,
        )["train"]

    def load_format_split(
        self,
        seed: int,
    ) -> tuple[Dataset, Dataset]:
        original_dataset = self.load()
        print(f"Original dataset size: {len(original_dataset):,}")
        data_subset = self.get_data_subset(
            Dataset.from_generator(self.format_dataset, gen_kwargs={"data": original_dataset}),
            seed,
        )
        train_data, test_data = extract_train_test(data_subset, test_size=TEST_FRACTION_OF_TRAIN, seed=seed)
        train_size = len(train_data)
        test_size = len(test_data)
        print(
            f"Cleaned and formatted data subset with train/test/total sizes: "
            f"{train_size:,}/{test_size:,}/{train_size + test_size:,}"
        )
        return train_data, test_data

from datasets import Dataset


def extract_train_test(
    dataset: Dataset,
    test_size: float,
    seed: int,
) -> tuple[Dataset, Dataset]:
    dataset_split = dataset.train_test_split(
        test_size=test_size,
        seed=seed,
    )
    return dataset_split["train"], dataset_split["test"]

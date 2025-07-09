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


def extract_prompt_and_response_from_chatml_pair(conversation: list[dict[str, str]]) -> tuple[str, str]:
    user_text, assistant_text = conversation
    assert user_text["role"] == "user"
    assert assistant_text["role"] == "assistant"
    return user_text["content"], assistant_text["content"]

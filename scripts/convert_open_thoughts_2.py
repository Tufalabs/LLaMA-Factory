import re
from pathlib import Path

from datasets import Dataset, load_dataset


SYSTEM = (
    "You are an assistant that thoroughly explores questions through a systematic long thinking process "
    "before providing the final precise and accurate solutions. "
    "This requires engaging in a comprehensive cycle of analysis, summarization, exploration, reassessment, "
    "reflection, backtracing, and iteration to develop a well-considered thinking process. "
    "Detail your reasoning process using the specified format: <think>thought with steps separated by '\n\n'</think> "
    "Each step should include detailed considerations such as analyzing questions, summarizing relevant findings, "
    "brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, "
    "and revisiting previous steps. Based on various attempts, explorations, and reflections from the thoughts, "
    "you should systematically present the final solution that you deem correct. "
    "The solution should remain a logical, accurate, concise expression style and detail necessary steps needed to "
    "reach the conclusion. Now, try to solve the following question through the above guidelines."
)
THINK_RE = re.compile(r"^<think>(.*)</think>(.*)$", re.DOTALL)
TRAIN_FRACTION = 0.1
TEST_FRACTION_OF_TRAIN = 0.02
SEED = 42


class InvalidResponse(ValueError):
    """Improper response formatting."""


def generate_data(data):
    for sample in data:
        prompt, response = extract_prompt_and_response(sample["conversations"])
        try:
            response = format_response(response)
        except InvalidResponse:
            continue

        yield {
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ],
            "source": sample["source"],
        }


def extract_prompt_and_response(conversation):
    user_text, assistant_text = conversation
    assert user_text["from"] == "user"
    assert assistant_text["from"] == "assistant"
    return user_text["value"], assistant_text["value"]


def format_response(response):
    match = THINK_RE.fullmatch(response)
    if not match:
        raise InvalidResponse(f"Invalid response:\n{response}")

    return f"<think>\n{match.group(1).strip()}\n</think>\n\n{match.group(2).strip()}"


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


def main():
    dataset_name = "OpenThoughts2-1M"
    original_dataset = load_dataset(
        f"open-thoughts/{dataset_name}",
        name="default",
        split="train",
    )
    print(f"Original dataset size: {len(original_dataset):,}")
    data_subset = Dataset.from_generator(generate_data, gen_kwargs={"data": original_dataset}).train_test_split(
        train_size=TRAIN_FRACTION,
        seed=SEED,
    )
    train_data, test_data = extract_train_test(data_subset["train"], test_size=TEST_FRACTION_OF_TRAIN, seed=SEED)
    train_size = len(train_data)
    test_size = len(test_data)
    print(
        f"Cleaned and formatted data subset with train/test/total sizes: "
        f"{train_size:,}/{test_size:,}/{train_size + test_size:,}"
    )
    train_save_path = Path(f"{dataset_name}-formatted-train.parquet").absolute()
    test_save_path = Path(f"{dataset_name}-formatted-test.parquet").absolute()
    train_data.to_parquet(train_save_path)
    print(f"Saved train dataset to: {train_save_path}")
    test_data.to_parquet(test_save_path)
    print(f"Saved test dataset to: {test_save_path}")


if __name__ == "__main__":
    main()

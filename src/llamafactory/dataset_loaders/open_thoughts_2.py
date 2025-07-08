import re
from collections.abc import Generator, Iterable
from typing import Any

from datasets import Dataset, load_dataset

from .base import BaseDatasetLoader
from .dataset_factory import DatasetFactory
from .names import DatasetName


NAME = DatasetName.OPEN_THOUGHTS_2
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


class MalformedTextError(ValueError):
    """Improper response formatting."""


@DatasetFactory.register_loader(NAME)
class OpenThoughts2Loader(BaseDatasetLoader):
    @classmethod
    def get_full_subset_size(cls) -> int | float:
        return 0.1

    @classmethod
    def load(cls) -> Dataset:
        return load_dataset(
            NAME.full_path,
            name="default",
            split="train",
        )

    @classmethod
    def format_dataset(cls, data: Iterable[dict[str, Any]]) -> Generator[dict[str, Any], None, None]:
        for sample in data:
            prompt, response = extract_prompt_and_response(sample["conversations"])
            try:
                response = format_response(response)
            except MalformedTextError:
                continue

            yield {
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ],
                "source": sample["source"],
            }


def extract_prompt_and_response(conversation: list[dict[str, str]]) -> tuple[str, str]:
    user_text, assistant_text = conversation
    assert user_text["from"] == "user"
    assert assistant_text["from"] == "assistant"
    return user_text["value"], assistant_text["value"]


def format_response(response: str) -> str:
    match = THINK_RE.fullmatch(response)
    if not match:
        raise MalformedTextError(f"Invalid response:\n{response}")

    return f"<think>\n{match.group(1).strip()}\n</think>\n\n{match.group(2).strip()}"

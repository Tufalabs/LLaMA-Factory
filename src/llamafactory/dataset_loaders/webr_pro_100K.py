from collections.abc import Generator, Iterable
from typing import Any

from datasets import Dataset, load_dataset

from .base import BaseDatasetLoader
from .dataset_factory import DatasetFactory
from .names import DatasetName
from .prompts import NEUTRAL_SYSTEM_PROMPT
from .utils import extract_prompt_and_response_from_chatml_pair


NAME = DatasetName.WEBR_PRO


@DatasetFactory.register_loader(NAME)
class WebRProLoader(BaseDatasetLoader):
    @classmethod
    def get_full_subset_size(cls) -> int | float:
        return 100_000

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
            prompt, response = extract_prompt_and_response_from_chatml_pair(sample["messages"])
            yield {
                "messages": [
                    {"role": "system", "content": NEUTRAL_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ],
                "prmopt_id": sample["prompt_id"],
            }

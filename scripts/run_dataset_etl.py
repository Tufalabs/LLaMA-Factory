import argparse
from pathlib import Path

from pydantic import BaseModel, field_validator

from llamafactory.dataset_loaders import DatasetFactory, DatasetName


DEFAULT_SEED = 42


class UserArgs(BaseModel):
    dataset: DatasetName
    save_path: Path
    seed: int

    @field_validator("dataset", mode="before")
    @classmethod
    def _validate_dataset(cls, val: DatasetName | str) -> DatasetName:
        if isinstance(val, DatasetName):
            return val

        try:
            return DatasetName(val)
        except ValueError:
            pass

        try:
            return DatasetName[val.upper()]
        except KeyError:
            pass

        raise ValueError(f"Unrecognized dataset '{val}'")

    @field_validator("save_path")
    @classmethod
    def _validate_save_path(cls, val: Path) -> Path:
        val = val.absolute()
        if not val.is_dir():
            raise ValueError(f"save-path doesn't exist: {val}")

        return val

    @staticmethod
    def from_argparse() -> "UserArgs":
        parser = argparse.ArgumentParser(description="Download and format ")
        parser.add_argument(
            "dataset",
            type=str,
        )
        parser.add_argument("-p", "--save-path", type=Path, default=".")
        parser.add_argument("-s", "--seed", type=int, default=DEFAULT_SEED)
        user_args = parser.parse_args()
        return UserArgs(dataset=user_args.dataset, save_path=user_args.save_path, seed=user_args.seed)


def main(args: UserArgs) -> None:
    loader = DatasetFactory.get_loader(args.dataset)
    train_data, test_data = loader.load_format_split(args.seed)
    train_save_path = args.save_path / f"{args.dataset.dataset_name}-formatted-train.parquet"
    test_save_path = args.save_path / f"{args.dataset.dataset_name}-formatted-test.parquet"
    train_data.to_parquet(train_save_path)
    print(f"Saved train dataset to: {train_save_path}")
    test_data.to_parquet(test_save_path)
    print(f"Saved test dataset to: {test_save_path}")


if __name__ == "__main__":
    main(UserArgs.from_argparse())

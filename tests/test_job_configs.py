import json
from pathlib import Path

import pytest
import yaml
from huggingface_hub import list_repo_refs


JOB_CONFIGS = list(Path("configs/").glob("*.yaml"))


@pytest.fixture
def dataset_info():
    with open("data/dataset_info.json") as f:
        return json.load(f)


def extract_dataset_name(raw_name: str) -> str:
    return "-".join(raw_name.split("_")[:-1])


@pytest.mark.tufa
@pytest.mark.parametrize("config_file", JOB_CONFIGS)
def test_model_names(config_file: Path) -> None:
    with open(config_file) as f:
        config = yaml.safe_load(f)

    expected_model_name = config_file.stem.split("_")[0]
    actual_model_name = config["model_name_or_path"].split("/")[-1]
    assert actual_model_name.lower() == expected_model_name.lower()
    assert config["run_name"].lower() == expected_model_name.lower()
    assert Path(config["output_dir"]).name.lower() == config_file.stem.lower()


@pytest.mark.tufa
@pytest.mark.parametrize("config_file", JOB_CONFIGS)
def test_dataset_names(config_file: Path) -> None:
    with open(config_file) as f:
        config = yaml.safe_load(f)

    expected_dataset_name = config_file.stem.split("_")[-1]
    assert config["dataset"].endswith("_train")
    assert config["eval_dataset"].endswith("_test")
    train_dataset_name = extract_dataset_name(config["dataset"])
    eval_dataset_name = extract_dataset_name(config["eval_dataset"])
    assert train_dataset_name == expected_dataset_name
    assert eval_dataset_name == expected_dataset_name


@pytest.mark.tufa
@pytest.mark.parametrize("config_file", JOB_CONFIGS)
def test_model_revisions(config_file: Path) -> None:
    with open(config_file) as f:
        config = yaml.safe_load(f)

    rev = config["model_revision"]
    if rev == "NA":
        return

    model_path = config["model_name_or_path"]
    branch_names = [b.name for b in list_repo_refs(model_path).branches]
    assert rev in branch_names


@pytest.mark.tufa
@pytest.mark.parametrize("config_file", JOB_CONFIGS)
def test_datasets_registered(config_file: Path, dataset_info) -> None:
    with open(config_file) as f:
        config = yaml.safe_load(f)

    registered_datasets = sorted(dataset_info.keys())
    assert config["dataset"] in registered_datasets
    assert config["eval_dataset"] in registered_datasets

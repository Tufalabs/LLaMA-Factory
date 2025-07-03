from pathlib import Path

import pytest
import yaml
from huggingface_hub import list_repo_refs


JOB_CONFIGS = list(Path("configs/").glob("*.yaml"))


@pytest.mark.parametrize("config_file", JOB_CONFIGS)
def test_model_names(config_file: Path) -> None:
    with open(config_file) as f:
        config = yaml.safe_load(f)

    expected_model_name = config_file.stem.split("_")[0]
    actual_model_name = config["model_name_or_path"].split("/")[-1]
    assert actual_model_name.lower() == expected_model_name.lower()
    assert config["run_name"].lower() == expected_model_name.lower()
    assert Path(config["output_dir"]).name.lower() == config_file.stem.lower()


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

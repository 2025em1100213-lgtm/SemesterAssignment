"""Shared pytest fixtures.

Creates a small synthetic dataset and trains artifacts once per test
session, entirely inside an isolated temp directory, so API tests can run
against a real loaded model WITHOUT overwriting the real (Telco-trained)
models/, data/, and artifacts/ committed in the repo.
"""

import os
import shutil
import sys
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.data import generate_synthetic_customers
from scripts.train import main as train_main

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(scope="session", autouse=True)
def trained_model_artifacts(tmp_path_factory):
    """Generate a small synthetic dataset and run the real training
    pipeline once per test session, inside a throwaway temp working
    directory, so tests are self-contained and never touch the real
    Telco-trained artifacts checked into the repo."""
    tmp_dir = tmp_path_factory.mktemp("test_run")

    os.makedirs(tmp_dir / "data" / "processed", exist_ok=True)
    os.makedirs(tmp_dir / "models", exist_ok=True)
    os.makedirs(tmp_dir / "artifacts" / "eval", exist_ok=True)
    os.makedirs(tmp_dir / "configs", exist_ok=True)
    shutil.copy(
        os.path.join(REPO_ROOT, "configs", "train_config.yaml"),
        tmp_dir / "configs" / "train_config.yaml",
    )

    df = generate_synthetic_customers(n_rows=800, start_date="2024-01-01", seed=123)
    df.to_csv(tmp_dir / "data" / "processed" / "training_data.csv", index=False)

    original_cwd = os.getcwd()
    os.chdir(tmp_dir)
    try:
        train_main("configs/train_config.yaml")
        # serving/main.py reads MODEL_PATH etc. as paths relative to cwd,
        # so keep cwd pointed at tmp_dir for the rest of the test session.
        yield
    finally:
        os.chdir(original_cwd)

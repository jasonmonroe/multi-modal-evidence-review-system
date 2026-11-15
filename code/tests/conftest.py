# code/tests/conftest.py
"""
Shared fixtures. Nothing here touches the network, the real dataset or the
real log file: all paths are redirected to `tmp_path` and the OpenAI client is
always replaced by a mock.
"""

import importlib
import sys
import types
from unittest.mock import MagicMock

import pandas as pd
import pytest


@pytest.fixture
def claims_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "user_id": "user_001",
                "image_paths": "images/test/case_001/img_1.jpg;images/test/case_001/img_2.jpg",
                "user_claim": "Customer: My bumper is dented. | Agent: Which side? | Customer: Front.",
                "claim_object": "car",
            },
            {
                "user_id": "user_002",
                "image_paths": "images/test/case_002/img_1.jpg",
                "user_claim": "Customer: Laptop screen cracked.",
                "claim_object": "laptop",
            },
        ]
    )


@pytest.fixture
def user_history_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"user_id": "user_001", "past_claim_count": 2, "history_flags": "none"},
            {
                "user_id": "user_002",
                "past_claim_count": 9,
                "history_flags": "user_history_risk",
            },
        ]
    )


@pytest.fixture
def evidence_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "requirement_id": "REQ_ALL",
                "claim_object": "all",
                "applies_to": "general claim review",
                "minimum_image_evidence": "Object must be visible.",
            },
            {
                "requirement_id": "REQ_CAR",
                "claim_object": "car",
                "applies_to": "dent or scratch",
                "minimum_image_evidence": "Panel visible from an angle.",
            },
            {
                "requirement_id": "REQ_LAPTOP",
                "claim_object": "laptop",
                "applies_to": "screen",
                "minimum_image_evidence": "Screen visible and lit.",
            },
        ]
    )


@pytest.fixture
def fake_data(claims_df, user_history_df, evidence_df):
    """Stand-in for DataHandler (only the attributes the agent reads)."""
    return types.SimpleNamespace(
        claims=claims_df, user_history=user_history_df, evidence_reqs=evidence_df
    )


@pytest.fixture
def fake_image():
    """Stand-in for ImageHandler that records load() calls."""
    image = MagicMock()
    image.images = ["data:image/jpeg;base64,AAAA", "data:image/jpeg;base64,BBBB"]
    image.filenames = ["img_1.jpg", "img_2.jpg"]
    return image


@pytest.fixture
def clean_imports(monkeypatch):
    """Re-import the pipeline and main fresh for each test."""
    monkeypatch.delitem(sys.modules, "pipelines.run_claims_pipeline", raising=False)
    monkeypatch.delitem(sys.modules, "main", raising=False)
    yield
    sys.modules.pop("pipelines.run_claims_pipeline", None)
    sys.modules.pop("main", None)


@pytest.fixture
def pipeline_module(clean_imports):
    return importlib.import_module("pipelines.run_claims_pipeline")


@pytest.fixture
def main_module(clean_imports, monkeypatch):
    # Never read a developer's real ../.env during tests.
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    return importlib.import_module("main")

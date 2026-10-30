# code/tests/pipelines/test_run_claims_pipeline.py

import importlib
import sys
import types
from unittest.mock import MagicMock

import openai
import pytest

import src.claim_handler as claim_handler_mod
from src.constants import (
    CSV_HEADERS,
    MAX_RETRIES,
    PAUSE_TIMER,
    SLEEP_TIMER,
    SLEEP_TIMER_INC,
)


def rate_limit_error():
    # Duck-typed response keeps the test independent of the HTTP client the
    # installed openai version happens to use.
    response = MagicMock(status_code=429, headers={})
    return openai.RateLimitError("slow down", response=response, body=None)


def result_for(idx, **over):
    row = {
        "user_id": f"user_{idx}",
        "image_paths": ["images/a.jpg"],
        "user_claim": f"claim {idx}",
        "claim_object": "car",
        "evidence_standard_met": True,
        "evidence_standard_met_reason": "ok",
        "risk_flags": "none",
        "issue_type": "dent",
        "object_part": "door",
        "claim_status": "supported",
        "claim_status_justification": "seen",
        "supporting_image_ids": ["img_1.jpg"],
        "valid_image": True,
        "severity": "low",
        "usage": {"total_tokens": 1},  # extra key that must not leak to CSV
    }
    row.update(over)
    return row


class FakeAnalyzer:
    def __init__(self, data, image, side_effects=None):
        self.data = data
        self.calls = []
        self.side_effects = list(side_effects or [])

    def analyze_claim(self, idx, strategy):
        self.calls.append((idx, strategy))
        if self.side_effects:
            effect = self.side_effects.pop(0)
            if isinstance(effect, Exception):
                raise effect
            return effect
        return result_for(idx)


@pytest.fixture
def harness(pipeline_module, monkeypatch, tmp_path, claims_df):
    out_file = tmp_path / "output.csv"
    monkeypatch.setattr(claim_handler_mod, "OUTPUT_FILE", str(out_file))

    sleeps = []
    monkeypatch.setattr(pipeline_module, "sleep", sleeps.append)

    holder = {}

    def make_analyzer(data, image):
        holder["analyzer"] = FakeAnalyzer(data, image, holder.get("effects"))
        return holder["analyzer"]

    monkeypatch.setattr(pipeline_module, "ClaimAnalyzer", make_analyzer)

    data = types.SimpleNamespace(claims=claims_df)
    return types.SimpleNamespace(
        mod=pipeline_module, data=data, out=out_file, sleeps=sleeps, holder=holder
    )


def read_rows(path):
    import csv

    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.reader(f))


class TestRunClaimsPipeline:
    def test_writes_header_plus_one_row_per_claim(self, harness):
        harness.mod.run_claims_pipeline(harness.data, MagicMock())
        rows = read_rows(harness.out)
        assert rows[0] == CSV_HEADERS
        assert len(rows) == 1 + len(harness.data.claims)
        assert all(len(r) == len(CSV_HEADERS) for r in rows)

    def test_analyzes_every_row_with_basic_prompt(self, harness):
        harness.mod.run_claims_pipeline(harness.data, MagicMock())
        assert harness.holder["analyzer"].calls == [
            (0, "Basic Prompt"),
            (1, "Basic Prompt"),
        ]

    def test_row_values_formatted_for_csv(self, harness):
        harness.mod.run_claims_pipeline(harness.data, MagicMock())
        row = dict(zip(CSV_HEADERS, read_rows(harness.out)[1]))
        assert row["user_id"] == "user_0"
        assert row["image_paths"] == "images/a.jpg"
        assert row["supporting_image_ids"] == "img_1.jpg"
        assert row["evidence_standard_met"] == "true"
        assert row["valid_image"] == "true"
        assert "usage" not in row

    def test_pauses_after_each_row(self, harness):
        harness.mod.run_claims_pipeline(harness.data, MagicMock())
        assert harness.sleeps == [PAUSE_TIMER, PAUSE_TIMER]

    def test_rate_limit_is_retried_then_succeeds(self, harness):
        harness.holder["effects"] = [rate_limit_error(), result_for(0), result_for(1)]
        harness.mod.run_claims_pipeline(harness.data, MagicMock())

        # the failed attempt for row 0 is retried before moving on
        assert harness.holder["analyzer"].calls == [
            (0, "Basic Prompt"),
            (0, "Basic Prompt"),
            (1, "Basic Prompt"),
        ]
        assert len(read_rows(harness.out)) == 3

    def test_rate_limit_backoff_grows(self, harness):
        harness.holder["effects"] = [rate_limit_error(), result_for(0), result_for(1)]
        harness.mod.run_claims_pipeline(harness.data, MagicMock())
        extra = SLEEP_TIMER_INC
        assert harness.sleeps == [
            SLEEP_TIMER,  # back-off after the 429
            PAUSE_TIMER + extra,  # pacing now includes the extra time
            PAUSE_TIMER + extra,
        ]

    def test_gives_up_after_max_retries(self, harness):
        harness.holder["effects"] = [rate_limit_error() for _ in range(MAX_RETRIES + 1)]
        with pytest.raises(openai.RateLimitError):
            harness.mod.run_claims_pipeline(harness.data, MagicMock())
        assert len(harness.holder["analyzer"].calls) == MAX_RETRIES + 1
        assert harness.sleeps == [SLEEP_TIMER] * MAX_RETRIES  # no sleep after giving up
        assert not harness.out.exists()

    def test_retry_budget_resets_for_each_row(self, harness):
        # MAX_RETRIES failures on every row is allowed; only exceeding it aborts
        err = [rate_limit_error() for _ in range(MAX_RETRIES)]
        harness.holder["effects"] = err + [result_for(0)] + list(err) + [result_for(1)]
        harness.mod.run_claims_pipeline(harness.data, MagicMock())
        assert len(read_rows(harness.out)) == 3

    def test_non_rate_limit_errors_propagate(self, harness):
        harness.holder["effects"] = [ValueError("bad row")]
        with pytest.raises(ValueError):
            harness.mod.run_claims_pipeline(harness.data, MagicMock())
        assert not harness.out.exists()  # nothing half-written

    def test_empty_claims_writes_header_only(self, harness):
        harness.data.claims = harness.data.claims.iloc[0:0]
        harness.mod.run_claims_pipeline(harness.data, MagicMock())
        assert read_rows(harness.out) == [CSV_HEADERS]
        assert harness.sleeps == []

    def test_prints_progress(self, harness, capsys):
        harness.mod.run_claims_pipeline(harness.data, MagicMock())
        out = capsys.readouterr().out
        assert "row 1 of 2" in out and "row 2 of 2" in out
        assert "100.0%" in out


def test_pipeline_imports_resolve(clean_imports):
    module = importlib.import_module("pipelines.run_claims_pipeline")
    assert module.ClaimAnalyzer.__module__ == "agents.claim_analyzer_agent"
    assert module.ClaimHandler.__module__ == "src.claim_handler"

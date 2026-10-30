# code/tests/evaluation/test_main.py

import types
from unittest.mock import MagicMock

import pandas as pd
import pytest

from evaluation import main as ev


def metrics(**over):
    base = {
        "correct": 1,
        "total": 2,
        "total_tokens": 1500,
        "total_cost": 0.015,
        "duration": 2.5,
        "accuracy": 50.0,
    }
    base.update(over)
    return base


class TestGenerateReportTemplate:
    @pytest.fixture
    def report(self, tmp_path, monkeypatch):
        path = tmp_path / "report.md"
        monkeypatch.setattr(ev, "REPORT_FILE", str(path))
        monkeypatch.setattr(ev, "MODEL_NAME", "test-model")
        return path

    def test_writes_markdown_with_expected_sections(self, report):
        ev.generate_report_template({"Basic Prompt": metrics()})
        text = report.read_text()
        assert text.startswith("# Operational Evaluation Report")
        for heading in (
            "### Strategy Comparison",
            "## Operational Analysis",
            "## Final Strategy Choice",
        ):
            assert heading in text

    def test_row_per_strategy(self, report):
        ev.generate_report_template(
            {"Basic Prompt": metrics(), "Chain-of-Thought Prompt": metrics(accuracy=75)}
        )
        text = report.read_text()
        assert "| Basic Prompt | 50.00% | 2.50s | $0.01 |" in text
        assert "| Chain-of-Thought Prompt | 75.00% | 2.50s | $0.01 |" in text

    def test_token_and_cost_totals(self, report):
        ev.generate_report_template(
            {"A": metrics(total_tokens=1500, total_cost=1.0), "B": metrics(total_tokens=2500, total_cost=0.5)}
        )
        text = report.read_text()
        assert "**Total Token Usage:** 4,000" in text
        assert "**Approximate Cost:** $1.50" in text
        assert "**Model Used:** test-model" in text

    def test_empty_results_still_writes_zero_totals(self, report):
        ev.generate_report_template({})
        text = report.read_text()
        assert "**Total Token Usage:** 0" in text
        assert "**Approximate Cost:** $0.00" in text

    def test_missing_metric_key_raises(self, report):
        with pytest.raises(KeyError):
            ev.generate_report_template({"A": {"accuracy": 1.0}})

    def test_overwrites_previous_report(self, report):
        report.write_text("old")
        ev.generate_report_template({})
        assert "old" not in report.read_text()


class FakeAnalyzer:
    """Returns canned decisions in order; records how it was called."""

    instances = []

    def __init__(self, data, image, statuses=None):
        self.statuses = statuses
        self.calls = []
        FakeAnalyzer.instances.append(self)

    def analyze_claim(self, idx, strategy):
        self.calls.append((idx, strategy))
        return {
            "claim_status": FakeAnalyzer.statuses[idx],
            "usage": {"total_tokens": 100},
        }


class TestRunEvaluationReportPipeline:
    @pytest.fixture
    def env(self, monkeypatch):
        FakeAnalyzer.instances = []
        FakeAnalyzer.statuses = ["supported", "contradicted"]
        monkeypatch.setattr(ev, "ClaimAnalyzer", FakeAnalyzer)
        report = MagicMock()
        monkeypatch.setattr(ev, "generate_report_template", report)
        banner = MagicMock()
        monkeypatch.setattr(ev, "show_banner", banner)

        sample = pd.DataFrame(
            {
                "claim_status": ["supported", "supported"],
                "user_id": ["a", "b"],
            }
        )
        data = MagicMock()
        data.claims = sample
        return types.SimpleNamespace(data=data, report=report, banner=banner)

    def test_loads_sample_data(self, env):
        ev.run_evaluation_report_pipeline(env.data, MagicMock())
        env.data.load_data.assert_called_once_with({"sample": True})

    def test_runs_both_strategies_over_every_row(self, env):
        ev.run_evaluation_report_pipeline(env.data, MagicMock())
        assert [a.calls for a in FakeAnalyzer.instances] == [
            [(0, "Basic Prompt"), (1, "Basic Prompt")],
            [(0, "Chain-of-Thought Prompt"), (1, "Chain-of-Thought Prompt")],
        ]

    def test_accuracy_and_token_metrics(self, env):
        ev.run_evaluation_report_pipeline(env.data, MagicMock())
        results = env.report.call_args.args[0]

        assert list(results) == ["Basic Prompt", "Chain-of-Thought Prompt"]
        basic = results["Basic Prompt"]
        assert basic["correct"] == 1 and basic["total"] == 2
        assert basic["accuracy"] == pytest.approx(50.0)
        assert basic["total_tokens"] == 200
        assert basic["total_cost"] == pytest.approx(0.2 / 1000 * 10)
        assert basic["duration"] >= 0
        assert isinstance(basic["accuracy"], float)

    def test_prefers_expected_claim_status_column(self, env):
        env.data.claims = pd.DataFrame(
            {
                "expected_claim_status": ["supported", "contradicted"],
                "claim_status": ["wrong", "wrong"],
            }
        )
        ev.run_evaluation_report_pipeline(env.data, MagicMock())
        assert env.report.call_args.args[0]["Basic Prompt"]["accuracy"] == 100.0

    def test_empty_sample_has_zero_accuracy_not_zero_division(self, env):
        env.data.claims = pd.DataFrame({"claim_status": []})
        ev.run_evaluation_report_pipeline(env.data, MagicMock())
        basic = env.report.call_args.args[0]["Basic Prompt"]
        assert basic["accuracy"] == 0 and basic["total"] == 0

    def test_missing_usage_counts_as_zero_tokens(self, env, monkeypatch):
        class NoUsage(FakeAnalyzer):
            def analyze_claim(self, idx, strategy):
                return {"claim_status": "supported"}

        monkeypatch.setattr(ev, "ClaimAnalyzer", NoUsage)
        ev.run_evaluation_report_pipeline(env.data, MagicMock())
        assert env.report.call_args.args[0]["Basic Prompt"]["total_tokens"] == 0

    @pytest.mark.xfail(
        raises=KeyError,
        strict=True,
        reason="get(..., default) eagerly evaluates ['claim_status'] even when "
        "only expected_claim_status exists",
    )
    def test_works_with_only_expected_claim_status_column(self, env):
        env.data.claims = pd.DataFrame({"expected_claim_status": ["supported", "supported"]})
        ev.run_evaluation_report_pipeline(env.data, MagicMock())

    @pytest.mark.xfail(
        strict=True,
        reason="subtitles.extend(str) adds one character per list item",
    )
    def test_summary_subtitles_are_whole_lines(self, env):
        ev.run_evaluation_report_pipeline(env.data, MagicMock())
        subtitles = env.banner.call_args_list[-1].args[1]
        assert any(s.startswith("Strategy: Basic Prompt") for s in subtitles)

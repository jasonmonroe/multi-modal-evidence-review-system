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
            "## Strategy Comparison",
            "## Operational Analysis",
            "## Final Strategy Choice",
        ):
            assert heading in text

    def test_row_per_strategy(self, report):
        ev.generate_report_template(
            {"Basic Prompt": metrics(), "Chain-of-Thought Prompt": metrics(accuracy=75)}
        )
        text = report.read_text()
        assert "| Basic Prompt | 1 / 2 | 50.00% | 2.50s | 1.25s | $0.01 |" in text
        assert "| Chain-of-Thought Prompt | 1 / 2 | 75.00% | 2.50s | 1.25s | $0.01 |" in text

    def test_no_layout_indentation_leaks_into_markdown(self, report):
        ev.generate_report_template(
            {"A": metrics(), "B": metrics(accuracy=75)}
        )
        for line in report.read_text().splitlines():
            assert line == line.lstrip(), repr(line)

    def test_sample_count_reported(self, report):
        ev.generate_report_template({"A": metrics(total=20)})
        assert "(20 labeled claims)" in report.read_text()

    def test_no_placeholder_left_in_output(self, report):
        ev.generate_report_template({"A": metrics()})
        text = report.read_text()
        assert "[X]" not in text and "because..." not in text

    def test_model_calls_and_throughput(self, report):
        # 2 strategies x 2 rows = 4 calls in 60s total -> 4 req/min, 3000 tok/min
        ev.generate_report_template(
            {"A": metrics(duration=30.0), "B": metrics(duration=30.0)}
        )
        text = report.read_text()
        assert "**Model Calls:** 4" in text
        assert "4.0 requests/min" in text
        assert "3,000 tokens/min" in text

    def test_zero_duration_and_zero_samples_do_not_divide_by_zero(self, report):
        ev.generate_report_template(
            {"A": metrics(total=0, correct=0, duration=0.0, accuracy=0)}
        )
        text = report.read_text()
        assert "0.0 requests/min" in text and "0.00s" in text

    def test_pacing_settings_documented(self, report):
        ev.generate_report_template({"A": metrics()})
        text = report.read_text()
        assert f"{ev.SLEEP_TIMER}s and retries" in text
        assert f"{ev.PAUSE_TIMER}s pause" in text

    def test_choice_prefers_highest_accuracy(self, report):
        ev.generate_report_template(
            {"Basic Prompt": metrics(accuracy=50), "Chain-of-Thought Prompt": metrics(accuracy=80)}
        )
        text = report.read_text()
        assert "We chose **Chain-of-Thought Prompt** because it had the highest accuracy (80.00%)." in text

    def test_choice_breaks_accuracy_tie_on_cost(self, report):
        ev.generate_report_template(
            {"A": metrics(total_cost=2.0), "B": metrics(total_cost=1.0)}
        )
        text = report.read_text()
        assert "We chose **B**" in text and "lowest estimated cost" in text

    def test_choice_breaks_full_tie_on_runtime(self, report):
        ev.generate_report_template(
            {"A": metrics(duration=9.0), "B": metrics(duration=3.0)}
        )
        assert "We chose **B**" in report.read_text()

    def test_single_strategy_choice(self, report):
        ev.generate_report_template({"Only": metrics()})
        assert "only strategy evaluated" in report.read_text()

    def test_no_strategies_has_no_choice(self, report):
        ev.generate_report_template({})
        assert "No strategies were evaluated." in report.read_text()

    def test_token_and_cost_totals(self, report):
        ev.generate_report_template(
            {"A": metrics(total_tokens=1500, total_cost=1.0), "B": metrics(total_tokens=2500, total_cost=0.5)}
        )
        text = report.read_text()
        assert "**Total Token Usage:** 4,000" in text
        assert "**Approximate Cost:** $1.50" in text
        assert "flat $0.01 per 1,000 tokens" in text
        assert "**Model Used:** test-model" in text

    def test_empty_results_still_writes_zero_totals(self, report):
        ev.generate_report_template({})
        text = report.read_text()
        assert "**Total Token Usage:** 0" in text
        assert "**Approximate Cost:** $0.00" in text
        assert "**Model Calls:** 0" in text

    def test_missing_metric_key_raises(self, report):
        with pytest.raises(KeyError):
            ev.generate_report_template({"A": {"accuracy": 1.0}})

    def test_overwrites_previous_report(self, report):
        report.write_text("old")
        ev.generate_report_template({})
        assert "old" not in report.read_text()


BASELINE = {"label": "supported", "count": 12, "total": 20, "accuracy": 60.0}


class TestBaselineSection:
    @pytest.fixture
    def report(self, tmp_path, monkeypatch):
        path = tmp_path / "report.md"
        monkeypatch.setattr(ev, "REPORT_FILE", str(path))
        return path

    def test_below_baseline_is_stated_plainly(self, report):
        ev.generate_report_template({"A": metrics(accuracy=50.0)}, baseline=BASELINE)
        text = report.read_text()
        assert "## Baseline" in text
        assert "always answers `supported` scores 60.00% (12 of 20 claims)" in text
        assert "50.00%, which is 10.00 points below it" in text

    def test_above_baseline(self, report):
        ev.generate_report_template({"A": metrics(accuracy=75.0)}, baseline=BASELINE)
        assert "beats it by 15.00 percentage points" in report.read_text()

    def test_equal_to_baseline(self, report):
        ev.generate_report_template({"A": metrics(accuracy=60.0)}, baseline=BASELINE)
        assert "only matches it" in report.read_text()

    def test_no_baseline_no_section(self, report):
        ev.generate_report_template({"A": metrics()})
        assert "## Baseline" not in report.read_text()


class TestMajorityBaseline:
    def test_computes_majority_label(self):
        df = pd.DataFrame({"claim_status": ["supported"] * 3 + ["contradicted"]})
        out = ev._majority_baseline(df, "claim_status")
        assert out == {"label": "supported", "count": 3, "total": 4, "accuracy": 75.0}

    def test_empty_frame(self):
        assert ev._majority_baseline(pd.DataFrame({"claim_status": []}), "claim_status") is None

    def test_missing_column(self):
        assert ev._majority_baseline(pd.DataFrame({"x": [1]}), "claim_status") is None


class TestConfusionSection:
    @pytest.fixture
    def report(self, tmp_path, monkeypatch):
        path = tmp_path / "report.md"
        monkeypatch.setattr(ev, "REPORT_FILE", str(path))
        return path

    def test_matrix_rendered_per_strategy(self, report):
        confusion = {"supported": {"supported": 3, "contradicted": 1}, "contradicted": {"supported": 2}}
        ev.generate_report_template({"A": metrics(confusion=confusion)})
        text = report.read_text()
        assert "### Confusion Matrix: A" in text
        assert "| supported | 3 | 1 | 0 |" in text
        assert "| contradicted | 2 | 0 | 0 |" in text
        assert "| not_enough_information | 0 | 0 | 0 |" in text

    def test_unexpected_label_gets_its_own_column(self, report):
        ev.generate_report_template({"A": metrics(confusion={"supported": {None: 2}})})
        assert "None" in report.read_text()

    def test_omitted_without_confusion_data(self, report):
        ev.generate_report_template({"A": metrics()})
        assert "Confusion Matrix" not in report.read_text()


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

    def test_confusion_matrix_and_baseline_are_collected(self, env):
        ev.run_evaluation_report_pipeline(env.data, MagicMock())
        basic = env.report.call_args.args[0]["Basic Prompt"]
        assert basic["confusion"] == {"supported": {"supported": 1, "contradicted": 1}}
        baseline = env.report.call_args.kwargs["baseline"]
        assert baseline["label"] == "supported" and baseline["accuracy"] == 100.0

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

    def test_works_with_only_expected_claim_status_column(self, env):
        env.data.claims = pd.DataFrame({"expected_claim_status": ["supported", "supported"]})
        ev.run_evaluation_report_pipeline(env.data, MagicMock())
        assert env.report.call_args.args[0]["Basic Prompt"]["accuracy"] == 50.0

    def test_summary_subtitles_are_whole_lines(self, env):
        ev.run_evaluation_report_pipeline(env.data, MagicMock())
        subtitles = env.banner.call_args_list[-1].args[1]
        assert any(s.startswith("Strategy: Basic Prompt") for s in subtitles)

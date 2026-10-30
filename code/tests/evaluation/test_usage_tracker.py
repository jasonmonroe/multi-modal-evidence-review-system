# code/tests/evaluation/test_usage_tracker.py

import pytest

from evaluation.usage_tracker import ModelUsage, UsageTracker
from src.constants import MAX_TOKENS


class TestModelUsage:
    def test_defaults(self):
        usage = ModelUsage()
        assert (usage.calls, usage.input_tokens, usage.output_tokens) == (0, 0, 0)
        assert usage.response_id == ""
        assert usage.least == {"response_id": "", "tokens": MAX_TOKENS * 2, "prompt": ""}
        assert usage.most == {"response_id": "", "tokens": 0, "prompt": ""}

    def test_total_tokens(self):
        assert ModelUsage(input_tokens=7, output_tokens=5).total_tokens == 12

    def test_total_tokens_is_int(self):
        assert isinstance(ModelUsage().total_tokens, int)

    def test_default_dicts_not_shared_between_instances(self):
        a, b = ModelUsage(), ModelUsage()
        a.least["tokens"] = 1
        a.most["tokens"] = 99
        assert b.least["tokens"] == MAX_TOKENS * 2
        assert b.most["tokens"] == 0


class TestUsageTracker:
    def test_init_empty(self):
        tracker = UsageTracker()
        assert tracker.by_model() == {}

    def test_record_accumulates_per_model(self):
        t = UsageTracker()
        t.record("m1", 10, 5, "p1", "r1")
        t.record("m1", 20, 1, "p2", "r2")
        usage = t.by_model()["m1"]
        assert usage.calls == 2
        assert usage.input_tokens == 30
        assert usage.output_tokens == 6
        assert usage.total_tokens == 36

    def test_models_tracked_separately(self):
        t = UsageTracker()
        t.record("m1", 1, 1, "p", "r1")
        t.record("m2", 2, 2, "p", "r2")
        assert set(t.by_model()) == {"m1", "m2"}
        assert t.by_model()["m2"].total_tokens == 4

    def test_min_and_max_tracking(self):
        t = UsageTracker()
        t.record("m", 100, 0, "mid", "r1")
        t.record("m", 10, 0, "small", "r2")
        t.record("m", 500, 0, "big", "r3")
        t.record("m", 50, 0, "ignored", "r4")

        usage = t.by_model()["m"]
        assert usage.least["tokens"] == 10
        assert usage.least["prompt"] == "small"
        assert usage.least["resource_id"] == "r2"
        assert usage.most["tokens"] == 500
        assert usage.most["prompt"] == "big"
        assert usage.most["resource_id"] == "r3"

    def test_first_call_sets_both_extremes(self):
        t = UsageTracker()
        t.record("m", 3, 4, "only", "r1")
        usage = t.by_model()["m"]
        assert usage.least["tokens"] == usage.most["tokens"] == 7

    def test_record_zero_tokens(self):
        t = UsageTracker()
        t.record("m", 0, 0, "empty", "r1")
        usage = t.by_model()["m"]
        assert usage.calls == 1 and usage.total_tokens == 0
        assert usage.least["tokens"] == 0

    @pytest.mark.xfail(
        strict=True,
        reason="a first call above MAX_TOKENS*2 never replaces the default `least`",
    )
    def test_oversized_first_call_is_recorded_as_least(self):
        t = UsageTracker()
        t.record("m", MAX_TOKENS * 3, 0, "huge", "r1")
        assert t.by_model()["m"].least["prompt"] == "huge"

    def test_by_model_returns_a_copy(self):
        t = UsageTracker()
        t.record("m", 1, 1, "p", "r1")
        snapshot = t.by_model()
        snapshot.clear()
        assert "m" in t.by_model()

    def test_by_model_values_are_model_usage(self):
        t = UsageTracker()
        t.record("m", 1, 1, "p", "r1")
        assert all(isinstance(v, ModelUsage) for v in t.by_model().values())

    def test_totals_empty(self):
        totals = UsageTracker().totals()
        assert isinstance(totals, ModelUsage)
        assert totals.calls == 0 and totals.total_tokens == 0

    def test_totals_sums_across_models(self):
        t = UsageTracker()
        t.record("m1", 10, 5, "p", "r1")
        t.record("m1", 1, 1, "p", "r2")
        t.record("m2", 100, 50, "p", "r3")
        totals = t.totals()
        assert totals.calls == 3
        assert totals.input_tokens == 111
        assert totals.output_tokens == 56
        assert totals.total_tokens == 167

    def test_totals_does_not_mutate_per_model_data(self):
        t = UsageTracker()
        t.record("m1", 10, 5, "p", "r1")
        t.totals()
        assert t.by_model()["m1"].calls == 1

    def test_response_id_is_required(self):
        with pytest.raises(TypeError):
            UsageTracker().record("m", 1, 1, "p")

    def test_response_id_recorded_in_min_and_max(self):
        t = UsageTracker()
        t.record("m", 5, 5, "p", "resp_abc")
        usage = t.by_model()["m"]
        assert usage.least["resource_id"] == usage.most["resource_id"] == "resp_abc"

# code/tests/agents/test_claim_analyzer_agent.py

from unittest.mock import MagicMock

import pandas as pd
import pytest

from agents import claim_analyzer_agent as agent
from agents.claim_analyzer_agent import ClaimAnalyzer
from src.constants import DECISION_C, DECISION_N, DECISION_S


@pytest.fixture
def model_cls(monkeypatch):
    cls = MagicMock(name="InsuranceAgentModel")
    monkeypatch.setattr(agent, "InsuranceAgentModel", cls)
    return cls


@pytest.fixture
def no_log(monkeypatch):
    log = MagicMock()
    monkeypatch.setattr(agent, "log_chat_transcript", log)
    return log


@pytest.fixture
def analyzer(fake_data, fake_image, model_cls):
    return ClaimAnalyzer(fake_data, fake_image)


def llm_response(**over):
    base = {
        "claim_status": DECISION_S,
        "claim_status_justification": "Dent visible.",
        "evidence_standard_met": True,
        "evidence_standard_met_reason": "clear",
        "issue_type": "dent",
        "object_part": "front_bumper",
        "risk_flags": "none",
        "severity": "low",
        "supporting_image_ids": "img_1.jpg",
        "valid_image": True,
    }
    base.update(over)
    return base


class TestInit:
    def test_wires_dependencies(self, analyzer, fake_data, fake_image, model_cls):
        assert analyzer.data is fake_data
        assert analyzer.image is fake_image
        model_cls.assert_called_once_with(2)
        assert analyzer._model is model_cls.return_value
        assert isinstance(analyzer._row, pd.Series) and analyzer._row.empty


class TestGetByIndex:
    def test_returns_expected_keys_and_types(self, analyzer):
        out = analyzer.get_by_index(0)
        assert set(out) == {
            "user_id",
            "image_paths",
            "user_claim",
            "claim_object",
            "images",
            "image_filenames",
            "user_history",
            "csv_row",
        }
        assert out["user_id"] == "user_001"
        assert out["claim_object"] == "car"
        assert isinstance(out["user_claim"], str)
        assert isinstance(out["images"], list)
        assert isinstance(out["image_filenames"], list)
        assert isinstance(out["user_history"], pd.DataFrame)
        assert isinstance(out["csv_row"], pd.Series)

    def test_loads_images_from_semicolon_paths(self, analyzer, fake_image):
        analyzer.get_by_index(0)
        fake_image.load.assert_called_once_with(
            ["images/test/case_001/img_1.jpg", "images/test/case_001/img_2.jpg"]
        )

    def test_single_image_path(self, analyzer, fake_image):
        analyzer.get_by_index(1)
        fake_image.load.assert_called_once_with(["images/test/case_002/img_1.jpg"])

    def test_claim_pipes_become_newlines(self, analyzer):
        out = analyzer.get_by_index(0)
        assert " | " not in out["user_claim"]
        assert out["user_claim"].count("\n") == 2

    def test_history_filtered_to_user(self, analyzer):
        history = analyzer.get_by_index(1)["user_history"]
        assert list(history["user_id"]) == ["user_002"]

    def test_sets_current_row(self, analyzer):
        analyzer.get_by_index(1)
        assert analyzer._row["user_id"] == "user_002"

    def test_raw_image_paths_preserved(self, analyzer, claims_df):
        assert analyzer.get_by_index(0)["image_paths"] == claims_df.loc[0, "image_paths"]

    def test_out_of_range_index(self, analyzer):
        with pytest.raises(IndexError):
            analyzer.get_by_index(99)

    def test_missing_image_paths_column(self, analyzer, fake_data):
        fake_data.claims = fake_data.claims.drop(columns=["image_paths"])
        with pytest.raises(KeyError):
            analyzer.get_by_index(0)


class TestFormatUserClaim:
    def test_replaces_separator_and_strips(self, analyzer):
        assert analyzer._format_user_claim("  A: hi | B: yo  ") == "A: hi\nB: yo"

    def test_no_separator(self, analyzer):
        assert analyzer._format_user_claim("single") == "single"

    def test_pipe_without_spaces_is_kept(self, analyzer):
        assert analyzer._format_user_claim("a|b") == "a|b"

    def test_empty(self, analyzer):
        assert analyzer._format_user_claim("") == ""

    def test_non_string_raises(self, analyzer):
        with pytest.raises(AttributeError):
            analyzer._format_user_claim(None)


class TestGetUserHistory:
    def test_known_user(self, analyzer):
        out = analyzer._get_user_history("user_001")
        assert len(out) == 1 and out.iloc[0]["history_flags"] == "none"

    def test_unknown_user_gives_empty_frame_with_columns(self, analyzer):
        out = analyzer._get_user_history("ghost")
        assert isinstance(out, pd.DataFrame) and out.empty
        assert "history_flags" in out.columns

    def test_exact_match_only(self, analyzer):
        assert analyzer._get_user_history("user_00").empty


class TestGenerateEvidencePrompt:
    def prompt(self, analyzer, obj):
        analyzer._row = pd.Series({"claim_object": obj})
        return analyzer._generate_evidence_prompt()

    def test_includes_general_and_object_rules(self, analyzer):
        text = self.prompt(analyzer, "car")
        assert "REQ_ALL" in text and "REQ_CAR" in text
        assert "Object must be visible." in text
        assert "Panel visible from an angle." in text

    def test_excludes_other_object_rules(self, analyzer):
        text = self.prompt(analyzer, "car")
        assert "REQ_LAPTOP" not in text and "Screen visible" not in text

    def test_unknown_object_gets_only_general_rules(self, analyzer):
        text = self.prompt(analyzer, "boat")
        assert "REQ_ALL" in text
        assert "REQ_CAR" not in text and "REQ_LAPTOP" not in text

    def test_header_instructions_and_numbering(self, analyzer):
        text = self.prompt(analyzer, "car")
        assert text.startswith("You must enforce the following strict evidence")
        assert "`evidence_standard_met` to `false`" in text
        assert "2. [RULE: REQ_ALL]" in text
        assert "3. [RULE: REQ_CAR]" in text

    def test_each_rule_has_failure_condition(self, analyzer):
        text = self.prompt(analyzer, "laptop")
        assert text.count("FAILURE CONDITION") == 2
        assert text.count("MANDATE:") == 2

    def test_returns_stripped_string(self, analyzer):
        text = self.prompt(analyzer, "car")
        assert isinstance(text, str) and text == text.strip()

    def test_no_requirements_at_all(self, analyzer, fake_data):
        fake_data.evidence_reqs = fake_data.evidence_reqs.iloc[0:0]
        text = self.prompt(analyzer, "car")
        assert "[RULE:" not in text


class TestShowResponse:
    def test_prints_each_key_title_cased(self, analyzer, capsys, monkeypatch):
        monkeypatch.setattr(agent, "MODEL_NAME", "m-x")
        analyzer.show_response({"claim_status": "supported", "valid_image": True})
        out = capsys.readouterr().out
        assert "MODEL: M-X (RESPONSE)" in out
        assert "Claim Status: supported" in out
        assert "Valid Image: True" in out

    def test_empty_response_prints_only_title(self, analyzer, capsys):
        analyzer.show_response({})
        assert "(RESPONSE)" in capsys.readouterr().out


class TestCall:
    def test_updates_model_and_returns_response(self, analyzer, capsys):
        analyzer._model.get_response.return_value = llm_response()
        out = analyzer.call({"user_id": "u"})
        analyzer._model.update.assert_called_once_with({"user_id": "u"})
        assert out == llm_response()
        assert "Run Time" in capsys.readouterr().out

    def test_empty_dataset_skips_update(self, analyzer):
        analyzer._model.get_response.return_value = {}
        analyzer.call({})
        analyzer._model.update.assert_not_called()

    def test_dict_response_is_displayed(self, analyzer, monkeypatch):
        shown = MagicMock()
        monkeypatch.setattr(analyzer, "show_response", shown)
        analyzer._model.get_response.return_value = {"a": 1}
        analyzer.call({"x": 1})
        shown.assert_called_once_with({"a": 1})

    def test_non_dict_response_not_displayed(self, analyzer, monkeypatch):
        shown = MagicMock()
        monkeypatch.setattr(analyzer, "show_response", shown)
        analyzer._model.get_response.return_value = None
        assert analyzer.call({"x": 1}) is None
        shown.assert_not_called()


class TestAnalyzeClaim:
    def test_returns_merged_dataset(self, analyzer, no_log):
        analyzer._model.get_response.return_value = llm_response()
        out = analyzer.analyze_claim(0, "Basic Prompt")

        # original row data, derived prompt fields and LLM fields all present
        for key in ("user_id", "claim_object", "images", "csv_row", "user_history"):
            assert key in out
        assert out["strategy"] == "Basic Prompt"
        assert "REQ_ALL" in out["evidence"]
        assert out["claim_status"] == DECISION_S
        assert out["issue_type"] == "dent"

    def test_sends_dataset_with_evidence_to_model(self, analyzer, no_log):
        analyzer._model.get_response.return_value = llm_response()
        analyzer.analyze_claim(0, "S")
        sent = analyzer._model.update.call_args.args[0]
        assert "evidence" in sent and sent["strategy"] == "S"

    def test_logs_three_stages(self, analyzer, no_log):
        analyzer._model.get_response.return_value = llm_response()
        analyzer.analyze_claim(1, "S")
        stages = [c.args[0] for c in no_log.call_args_list]
        assert stages == [
            "ROW INDEX 1 - INPUT PAYLOAD",
            "ROW INDEX 1 - RAW LLM RESPONSE",
            "ROW INDEX 1 - POST-PROCESS MAKE DECISION",
        ]

    def test_decision_rules_applied_to_output(self, analyzer, no_log):
        # user_002 has user_history_risk in the fixture history
        analyzer._model.get_response.return_value = llm_response()
        out = analyzer.analyze_claim(1, "S")
        assert out["risk_flags"] == "manual_review_required;user_history_risk"

    def test_injection_flag_forces_contradicted(self, analyzer, no_log):
        analyzer._model.get_response.return_value = llm_response(
            risk_flags="text_instruction_present"
        )
        out = analyzer.analyze_claim(0, "S")
        assert out["claim_status"] == DECISION_C
        assert out["valid_image"] is False

    def test_empty_model_response_does_not_crash(self, analyzer, no_log):
        analyzer._model.get_response.return_value = {}
        out = analyzer.analyze_claim(0, "S")
        assert out["risk_flags"] == "none"
        assert out["claim_status"] is None
        assert out["evidence_standard_met"] is True

    def test_bad_index_propagates(self, analyzer, no_log):
        with pytest.raises(IndexError):
            analyzer.analyze_claim(50, "S")


class TestMakeDecision:
    def decide(self, analyzer, history_flag="none", **fields):
        dataset = llm_response()
        dataset.update(fields)
        dataset["user_history"] = pd.DataFrame({"history_flags": [history_flag]})
        return analyzer._make_decision(dataset)

    def test_clean_pass_through(self, analyzer):
        out = self.decide(analyzer)
        assert out["claim_status"] == DECISION_S
        assert out["risk_flags"] == "none"
        assert out["severity"] == "low"
        assert out["valid_image"] is True
        assert out["evidence_standard_met"] is True

    def test_output_types(self, analyzer):
        out = self.decide(analyzer, risk_flags="blurry_image;wrong_angle")
        assert isinstance(out["risk_flags"], str)
        assert isinstance(out["valid_image"], bool)
        assert isinstance(out["evidence_standard_met"], bool)
        assert isinstance(out["claim_status"], str)

    def test_returns_same_dataset_object_and_keeps_other_keys(self, analyzer):
        dataset = llm_response()
        dataset["user_history"] = pd.DataFrame({"history_flags": ["none"]})
        dataset["issue_type"] = "scratch"
        assert analyzer._make_decision(dataset) is dataset
        assert dataset["issue_type"] == "scratch"

    def test_flags_sorted_and_deduplicated(self, analyzer):
        out = self.decide(analyzer, risk_flags="wrong_angle;blurry_image;wrong_angle")
        assert out["risk_flags"] == "blurry_image;wrong_angle"

    def test_none_token_removed_from_mixed_flags(self, analyzer):
        out = self.decide(analyzer, risk_flags="none;blurry_image")
        assert out["risk_flags"] == "blurry_image"

    def test_history_risk_adds_flags(self, analyzer):
        out = self.decide(analyzer, history_flag="user_history_risk")
        assert out["risk_flags"] == "manual_review_required;user_history_risk"
        assert out["claim_status"] == DECISION_S  # flags alone don't change status

    def test_other_history_flag_ignored(self, analyzer):
        out = self.decide(analyzer, history_flag="some_other_flag")
        assert out["risk_flags"] == "none"

    def test_empty_history_ignored(self, analyzer):
        dataset = llm_response()
        dataset["user_history"] = pd.DataFrame({"history_flags": []})
        assert analyzer._make_decision(dataset)["risk_flags"] == "none"

    @pytest.mark.parametrize("flag", ["text_instruction_present", "possible_manipulation"])
    def test_injection_or_manipulation_contradicts(self, analyzer, flag):
        out = self.decide(analyzer, risk_flags=flag)
        assert out["claim_status"] == DECISION_C
        assert out["valid_image"] is False
        assert flag in out["risk_flags"]
        assert "manual_review_required" in out["risk_flags"]

    def test_non_original_image_invalidates_but_keeps_status(self, analyzer):
        out = self.decide(analyzer, risk_flags="non_original_image")
        assert out["valid_image"] is False
        assert out["claim_status"] == DECISION_S
        assert "manual_review_required" not in out["risk_flags"]

    def test_unrelated_flag_does_not_invalidate(self, analyzer):
        out = self.decide(analyzer, risk_flags="blurry_image")
        assert out["valid_image"] is True and out["claim_status"] == DECISION_S

    @pytest.mark.parametrize("severity", ["low", "medium", "high"])
    def test_not_enough_info_resets_concrete_severity(self, analyzer, severity):
        out = self.decide(analyzer, claim_status=DECISION_N, severity=severity)
        assert out["severity"] == "unknown"

    @pytest.mark.parametrize("severity", ["none", "unknown"])
    def test_not_enough_info_keeps_none_or_unknown_severity(self, analyzer, severity):
        out = self.decide(analyzer, claim_status=DECISION_N, severity=severity)
        assert out["severity"] == severity

    def test_severity_kept_for_supported(self, analyzer):
        assert self.decide(analyzer, severity="high")["severity"] == "high"

    def test_unmet_evidence_forces_not_enough_information(self, analyzer):
        out = self.decide(analyzer, evidence_standard_met=False)
        assert out["claim_status"] == DECISION_N
        assert out["evidence_standard_met"] is False

    def test_unmet_evidence_overrides_contradicted(self, analyzer):
        out = self.decide(
            analyzer, evidence_standard_met=False, risk_flags="possible_manipulation"
        )
        assert out["claim_status"] == DECISION_N

    @pytest.mark.xfail(
        strict=True,
        reason="severity reset runs before the evidence override, so a claim "
        "downgraded to not_enough_information keeps its concrete severity",
    )
    def test_downgraded_claim_has_unknown_severity(self, analyzer):
        out = self.decide(analyzer, evidence_standard_met=False, severity="high")
        assert out["severity"] == "unknown"

    def test_missing_fields_use_defaults(self, analyzer):
        out = analyzer._make_decision(
            {"user_history": pd.DataFrame({"history_flags": ["none"]})}
        )
        assert out["claim_status"] is None
        assert out["risk_flags"] == "none"
        assert out["severity"] == "unknown"
        assert out["valid_image"] is True
        assert out["evidence_standard_met"] is True

    def test_missing_history_raises(self, analyzer):
        with pytest.raises(TypeError):
            analyzer._make_decision({"risk_flags": "none"})

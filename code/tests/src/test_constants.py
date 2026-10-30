# code/tests/src/test_constants.py

import os
import re

from src import constants as c


class TestScalarConstants:
    def test_numeric_constants_are_positive_ints(self):
        for name in (
            "MAX_TOKENS",
            "TOKEN_UNIT",
            "SECS_IN_MIN",
            "PAUSE_TIMER",
            "SLEEP_TIMER",
            "SLEEP_TIMER_INC",
            "MAX_RETRIES",
            "MSEC",
            "PEP8_LINE_LEN",
        ):
            value = getattr(c, name)
            assert isinstance(value, int) and not isinstance(value, bool), name
            assert value > 0, name

    def test_known_values(self):
        assert c.SECS_IN_MIN == 60
        assert c.MSEC == 1000
        assert c.PEP8_LINE_LEN == 79

    def test_dataset_paths(self):
        assert c.DATASET_DIR.endswith("/")
        for path, name in (
            (c.CLAIMS_FILE, "claims.csv"),
            (c.SAMPLE_CLAIMS_FILE, "sample_claims.csv"),
            (c.EVIDENCE_REQ_FILE, "evidence_requirements.csv"),
            (c.USER_HISTORY_FILE, "user_history.csv"),
            (c.OUTPUT_FILE, "output.csv"),
        ):
            assert isinstance(path, str)
            assert path.startswith(c.DATASET_DIR)
            assert path.endswith(name)

    def test_paths_are_absolute_and_independent_of_cwd(self, tmp_path, monkeypatch):
        import importlib

        monkeypatch.chdir(tmp_path)
        reloaded = importlib.reload(c)
        try:
            for path in (
                reloaded.DATASET_DIR,
                reloaded.CLAIMS_FILE,
                reloaded.OUTPUT_FILE,
                reloaded.REPORT_FILE,
                reloaded.CHAT_TRANSCRIPT_FILE,
            ):
                assert os.path.isabs(path), path
            assert os.path.isdir(reloaded.DATASET_DIR)
            assert os.path.isdir(os.path.dirname(reloaded.REPORT_FILE))
        finally:
            monkeypatch.undo()
            importlib.reload(c)

    def test_decisions_are_distinct_strings(self):
        decisions = {c.DECISION_S, c.DECISION_C, c.DECISION_N}
        assert decisions == {"supported", "contradicted", "not_enough_information"}


class TestCsvHeaders:
    def test_headers_unique_and_strings(self):
        assert all(isinstance(h, str) for h in c.CSV_HEADERS)
        assert len(c.CSV_HEADERS) == len(set(c.CSV_HEADERS))

    def test_headers_cover_every_schema_field(self):
        assert set(c.RESPONSE_SCHEMA["properties"]) <= set(c.CSV_HEADERS)

    def test_headers_include_input_columns(self):
        for col in ("user_id", "image_paths", "user_claim", "claim_object"):
            assert col in c.CSV_HEADERS

    def test_header_count(self):
        assert len(c.CSV_HEADERS) == 14


class TestResponseSchema:
    schema = c.RESPONSE_SCHEMA

    def test_top_level_shape(self):
        assert self.schema["type"] == "object"
        assert self.schema["additionalProperties"] is False

    def test_required_matches_properties(self):
        # strict structured outputs demand every property be required
        assert set(self.schema["required"]) == set(self.schema["properties"])

    def test_property_types(self):
        props = self.schema["properties"]
        assert props["evidence_standard_met"]["type"] == "boolean"
        assert props["valid_image"]["type"] == "boolean"
        for name in (
            "claim_status",
            "claim_status_justification",
            "evidence_standard_met_reason",
            "issue_type",
            "object_part",
            "risk_flags",
            "severity",
            "supporting_image_ids",
        ):
            assert props[name]["type"] == "string", name

    def test_claim_status_enum(self):
        assert set(self.schema["properties"]["claim_status"]["enum"]) == {
            c.DECISION_S,
            c.DECISION_C,
            c.DECISION_N,
        }

    def test_severity_enum(self):
        assert self.schema["properties"]["severity"]["enum"] == [
            "none",
            "low",
            "medium",
            "high",
            "unknown",
        ]

    def test_enums_have_no_duplicates(self):
        for name in ("issue_type", "object_part", "claim_status", "severity"):
            enum = self.schema["properties"][name]["enum"]
            assert len(enum) == len(set(enum)), name

    def test_issue_type_contains_fallbacks(self):
        enum = self.schema["properties"]["issue_type"]["enum"]
        assert "none" in enum and "unknown" in enum
        assert "bogus" not in enum

    def test_object_parts_for_each_claim_object(self):
        enum = set(self.schema["properties"]["object_part"]["enum"])
        assert {"front_bumper", "windshield", "headlight"} <= enum  # car
        assert {"screen", "keyboard", "hinge"} <= enum  # laptop
        assert {"box", "seal", "label"} <= enum  # package
        assert "unknown" in enum


class TestRiskFlagPattern:
    pattern = re.compile(c.RESPONSE_SCHEMA["properties"]["risk_flags"]["pattern"])

    def test_accepts_single_flags(self):
        for flag in ("none", "blurry_image", "manual_review_required"):
            assert self.pattern.match(flag), flag

    def test_accepts_semicolon_joined_flags(self):
        assert self.pattern.match("blurry_image;wrong_angle;user_history_risk")

    def test_rejects_unknown_flag(self):
        assert not self.pattern.match("not_a_flag")

    def test_rejects_spaces_and_commas(self):
        assert not self.pattern.match("blurry_image; wrong_angle")
        assert not self.pattern.match("blurry_image,wrong_angle")

    def test_rejects_empty_and_trailing_separator(self):
        assert not self.pattern.match("")
        assert not self.pattern.match("blurry_image;")

    def test_rejects_prefix_only_match(self):
        # "none" must not match as a prefix of a longer bad token
        assert not self.pattern.match("none_of_the_above")


class TestPrompts:
    def test_developer_prompt_formats_with_model_name_only(self):
        rendered = c.DEVELOPER_PROMPT_INSTRUCTIONS.format(model_name="gpt-test")
        assert "gpt-test" in rendered
        assert "{model_name}" not in rendered

    def test_developer_prompt_lists_every_decision(self):
        for decision in (c.DECISION_S, c.DECISION_C, c.DECISION_N):
            assert decision in c.DEVELOPER_PROMPT_INSTRUCTIONS

    def test_developer_prompt_warns_about_prompt_injection(self):
        assert "prompt injection" in c.DEVELOPER_PROMPT_INSTRUCTIONS.lower()

    def test_payload_template_placeholders(self):
        fields = set(re.findall(r"{(\w+)}", c.USER_DATA_PAYLOAD_TEMPLATE))
        assert fields == {"claim_object", "user_claim", "images", "evidence"}

    def test_payload_template_renders(self):
        text = c.USER_DATA_PAYLOAD_TEMPLATE.format(
            claim_object="car", user_claim="hi", images="a.jpg", evidence="rules"
        )
        assert "car" in text and "a.jpg" in text and "rules" in text

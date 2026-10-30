# code/tests/models/test_insurance_agent_model.py

import json
import types
from unittest.mock import MagicMock

import pytest

from evaluation.usage_tracker import UsageTracker
from models import insurance_agent_model as iam
from src.constants import MAX_TOKENS, RESPONSE_SCHEMA


@pytest.fixture
def creds(monkeypatch):
    monkeypatch.setattr(iam, "MODEL_API_URL", "http://llm.test/v1")
    monkeypatch.setattr(iam, "MODEL_NAME", "test-model")
    monkeypatch.setattr(iam, "MODEL_API_KEY", "sk-test")


@pytest.fixture
def openai_cls(monkeypatch):
    cls = MagicMock(name="OpenAI")
    monkeypatch.setattr(iam, "OpenAI", cls)
    return cls


@pytest.fixture
def real_model(creds, openai_cls):
    return iam.InsuranceAgentModel(2)


@pytest.fixture
def model(real_model):
    # Isolate the model's own behaviour from the tracker; the two are covered
    # together by test_usage_is_recorded_with_real_tracker.
    real_model.usage = MagicMock(name="UsageTracker")
    return real_model


def completion(content='{"claim_status": "supported"}', usage=True, model_name="test-model"):
    """Chat completion stand-in. `prompt_token` mirrors the attribute the
    source reads (see the xfail test for the real field name)."""
    resp = MagicMock()
    resp.id = "resp_1"
    resp.model = model_name
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    if usage:
        resp.usage.prompt_token = 11
        resp.usage.completion_tokens = 5
        resp.usage.model_dump.return_value = {"total_tokens": 16}
    else:
        resp.usage = None
    return resp


class TestInit:
    def test_with_credentials(self, real_model, openai_cls):
        model = real_model
        assert model.name == "Insurance Agent Adjuster Model"
        openai_cls.assert_called_once_with(
            base_url="http://llm.test/v1",
            api_key="sk-test",
            timeout=120,
            max_retries=3,
        )
        assert model._client is openai_cls.return_value
        assert model._images == [] and model._image_filenames == []
        assert model._user_id == model._claim_object == model._evidence == ""
        assert isinstance(model.usage, UsageTracker)
        assert model.usage.by_model() == {}

    def test_banner_shows_row_total(self, creds, openai_cls, capsys):
        iam.InsuranceAgentModel(7)
        out = capsys.readouterr().out
        assert "CVS ROWS: 7" in out and "test-model" in out

    @pytest.mark.parametrize("missing", ["MODEL_API_URL", "MODEL_NAME", "MODEL_API_KEY"])
    def test_missing_credentials_warns_and_skips_client(
        self, creds, openai_cls, monkeypatch, capsys, missing
    ):
        monkeypatch.setattr(iam, missing, None)
        m = iam.InsuranceAgentModel()
        assert "Credentials" in capsys.readouterr().out
        openai_cls.assert_not_called()
        assert not hasattr(m, "_client")

    @pytest.mark.xfail(
        raises=AttributeError,
        strict=True,
        reason="__init__ returns before setting _client, so get_response crashes",
    )
    def test_get_response_without_credentials_returns_empty(
        self, creds, openai_cls, monkeypatch
    ):
        monkeypatch.setattr(iam, "MODEL_API_KEY", None)
        assert iam.InsuranceAgentModel().get_response() == {}

    def test_default_row_total(self, creds, openai_cls, capsys):
        iam.InsuranceAgentModel()
        assert "CVS ROWS: 0" in capsys.readouterr().out


class TestUpdate:
    def test_sets_private_attributes(self, model):
        model.update(
            {
                "user_id": "u1",
                "claim_object": "car",
                "user_claim": "dented",
                "images": ["data:x"],
                "image_filenames": ["a.jpg"],
                "evidence": "rules",
            }
        )
        assert model._user_id == "u1"
        assert model._claim_object == "car"
        assert model._user_claim == "dented"
        assert model._images == ["data:x"]
        assert model._image_filenames == ["a.jpg"]
        assert model._evidence == "rules"

    def test_ignores_unknown_keys(self, model):
        model.update({"user_history": object(), "csv_row": 1, "strategy": "x"})
        assert not hasattr(model, "_user_history")
        assert not hasattr(model, "_strategy")

    def test_does_not_touch_public_attributes(self, model):
        usage = model.usage
        model.update({"usage": "hijack", "name": "hijack"})
        assert model.usage is usage and model.name != "hijack"


class TestGetMessage:
    @pytest.fixture
    def loaded(self, model):
        model.update(
            {
                "claim_object": "car",
                "user_claim": "Customer: dent on door",
                "images": ["data:image/jpeg;base64,AA", "data:image/jpeg;base64,BB"],
                "image_filenames": ["img_1.jpg", "img_2.jpg"],
                "evidence": "RULE-TEXT",
            }
        )
        return model

    def test_roles_and_types(self, loaded):
        messages = loaded._get_message()
        assert [m["role"] for m in messages] == ["developer", "user"]
        assert isinstance(messages[0]["content"], str)
        assert isinstance(messages[1]["content"], list)

    def test_developer_prompt_is_fully_formatted(self, loaded):
        content = loaded._get_message()[0]["content"]
        assert "test-model" in content
        assert "{model_name}" not in content

    def test_metadata_block(self, loaded):
        first = loaded._get_message()[1]["content"][0]
        assert first["type"] == "text"
        for token in ("car", "Customer: dent on door", "img_1.jpg, img_2.jpg", "RULE-TEXT"):
            assert token in first["text"]

    def test_images_interleaved_after_their_labels(self, loaded):
        parts = loaded._get_message()[1]["content"]
        assert [p["type"] for p in parts] == [
            "text",
            "text",
            "image_url",
            "text",
            "image_url",
        ]
        assert "img_1.jpg" in parts[1]["text"]
        assert parts[2]["image_url"] == {
            "url": "data:image/jpeg;base64,AA",
            "detail": "high",
        }
        assert "img_2.jpg" in parts[3]["text"]
        assert parts[4]["image_url"]["url"].endswith("BB")

    def test_no_images_labelled_none(self, model):
        model.update({"claim_object": "car", "user_claim": "x"})
        parts = model._get_message()[1]["content"]
        assert len(parts) == 1
        assert "none" in parts[0]["text"]

    def test_mismatched_lengths_truncate_to_shorter(self, loaded):
        loaded.update({"images": ["data:only-one"]})
        parts = loaded._get_message()[1]["content"]
        assert sum(p["type"] == "image_url" for p in parts) == 1


class TestResponseFormat:
    def test_json_schema(self, model):
        schema = model._get_json_schema()
        assert schema["name"] == "insurance_agent_adjuster_model"
        assert " " not in schema["name"]
        assert schema["strict"] is True
        assert schema["schema"] is RESPONSE_SCHEMA
        assert isinstance(schema["description"], str) and schema["description"]

    def test_response_format(self, model):
        fmt = model._get_response_format()
        assert fmt["type"] == "json_schema"
        assert fmt["json_schema"] == model._get_json_schema()


class TestGetResponse:
    def test_calls_api_with_expected_arguments(self, model):
        model._client.chat.completions.create.return_value = completion()
        model.update({"claim_object": "car", "user_claim": "x"})

        result = model.get_response()

        kwargs = model._client.chat.completions.create.call_args.kwargs
        assert kwargs["model"] == "test-model"
        assert kwargs["max_completion_tokens"] == MAX_TOKENS
        assert kwargs["temperature"] == 0.0
        assert kwargs["top_p"] == 1.0
        assert kwargs["timeout"] == 90.0
        assert kwargs["response_format"]["type"] == "json_schema"
        assert kwargs["messages"][0]["role"] == "developer"
        assert result["claim_status"] == "supported"

    def test_no_client_returns_empty_dict(self, model):
        model._client = None
        assert model.get_response() == {}
        assert isinstance(model.get_response(), dict)

    def test_records_usage_with_prompt(self, model):
        model._client.chat.completions.create.return_value = completion()
        model.update({"claim_object": "car", "user_claim": "x"})
        model.get_response()
        kwargs = model.usage.record.call_args.kwargs
        assert kwargs["response_id"] == "resp_1"
        assert kwargs["model"] == "test-model"
        assert kwargs["input_tokens"] == 11
        assert kwargs["output_tokens"] == 5
        assert isinstance(kwargs["prompt"], list)  # the user content payload

    def test_api_errors_propagate(self, model):
        model._client.chat.completions.create.side_effect = RuntimeError("boom")
        with pytest.raises(RuntimeError):
            model.get_response()


class TestFilterResponse:
    def test_parses_json_and_attaches_usage(self, model):
        out = model._filter_response(completion('{"severity": "low"}'), "p")
        assert out == {"severity": "low", "usage": {"total_tokens": 16}}

    def test_usage_is_recorded_with_real_tracker(self, real_model):
        real_model._filter_response(completion(), "p")
        most = real_model.usage.by_model()["test-model"].most
        assert most["resource_id"] == "resp_1"
        assert most["tokens"] == 16

    @pytest.mark.parametrize("content", ["", None])
    def test_empty_content_returns_empty_dict(self, model, content):
        assert model._filter_response(completion(content), "p") == {}

    def test_invalid_json_returns_empty_dict(self, model, capsys):
        assert model._filter_response(completion("{not json"), "p") == {}
        assert "Error occurred while parsing" in capsys.readouterr().out

    def test_no_choices_returns_empty_dict(self, model):
        resp = completion()
        resp.choices = []
        assert model._filter_response(resp, "p") == {}

    def test_missing_message_returns_empty_dict(self, model):
        resp = completion()
        resp.choices = [types.SimpleNamespace()]  # no .message
        assert model._filter_response(resp, "p") == {}

    def test_result_is_plain_dict_with_json_types(self, model):
        out = model._filter_response(
            completion(json.dumps({"valid_image": True, "supporting_image_ids": "none"})),
            "p",
        )
        assert out["valid_image"] is True
        assert isinstance(out["supporting_image_ids"], str)

    @pytest.mark.xfail(
        raises=AttributeError,
        strict=True,
        reason="reads usage.prompt_token; the OpenAI field is prompt_tokens",
    )
    def test_real_openai_usage_shape(self, real_model):
        model = real_model
        usage = types.SimpleNamespace(
            prompt_tokens=11,
            completion_tokens=5,
            model_dump=lambda: {"total_tokens": 16},
        )
        resp = types.SimpleNamespace(
            id="r",
            model="test-model",
            usage=usage,
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(content='{"a": 1}')
                )
            ],
        )
        assert model._filter_response(resp, "p")["usage"] == {"total_tokens": 16}

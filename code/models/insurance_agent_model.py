# code/models/insurance_agent_model.py

# +---------------------------------------------------------------------------+
# |                       INSURANCE AGENT ADJUSTER                            |
# +---------------------------------------------------------------------------+

# Python Libraries
import json

# Vendor Libraries
from evaluation.usage_tracker import UsageTracker
from openai import OpenAI
from openai.types.chat import ChatCompletion

# Local Libraries
from src.constants import (
    DEVELOPER_PROMPT_INSTRUCTIONS,
    MAX_TOKENS,
    MODEL_API_KEY,
    MODEL_API_URL,
    MODEL_NAME,
    RESPONSE_SCHEMA,
    USER_DATA_PAYLOAD_TEMPLATE,
)
from src.utils import show_banner


class InsuranceAgentModel:
    """
    Insurance agentic Agent tasked with identifying damage to an item.
    """

    def __init__(self, row_total: int = 0):
        self.name = "Insurance Agent Adjuster Model"
        self._client = None
        self.usage = UsageTracker()
        self._user_id = ""
        self._claim_object = ""
        self._user_claim = ""
        self._images = []
        self._image_filenames = []  # Store actual image filenames
        self._evidence = ""

        if MODEL_API_URL is None or MODEL_NAME is None or MODEL_API_KEY is None:
            print("⚠️ Credentials aren't properly being read. Check .env file. ⚠️")
            return

        self._client = self._init_llm_model(row_total)

    def _init_llm_model(self, row_total: int) -> OpenAI | None:

        subtitles = [
            f"🤖MODEL_NAME: {MODEL_NAME}",
            f"🌐️MODEL_API_URL: {MODEL_API_URL}",
            f"📄️CVS ROWS: {row_total}",
        ]
        show_banner(self.name.upper(), subtitles)

        return OpenAI(
            base_url=MODEL_API_URL,
            api_key=MODEL_API_KEY,
            timeout=120,  # ⏱️ Kill the connection if it hangs over 120 seconds
            max_retries=3,  # 🔄 Automatically back off and retry 3 times natively
        )

    def update(self, dataset: dict) -> None:
        """
        Updating dataset attributes.
        :param dataset:
        :return:
        """

        for attr_name, value in dataset.items():
            class_attr = "_" + attr_name
            if hasattr(self, class_attr):
                setattr(self, class_attr, value)

    def _get_message(self) -> list:
        """
        Interleaves structural metadata instructions with aligned filename-to-image indicators
        so the vision model knows exactly which filename belongs to which image layout.
        """

        # Create clear text list of identifiers to track matching inside the metadata layout
        image_labels = (
            ", ".join(self._image_filenames) if self._image_filenames else "none"
        )

        # Format metadata fields cleanly into the payload structural array
        metadata_text = USER_DATA_PAYLOAD_TEMPLATE.format(
            claim_object=self._claim_object,
            user_claim=self._user_claim,
            images=image_labels,
            evidence=self._evidence,
        )

        # Build dynamic array content payload for user role
        user_content_payload = [
            {
                "type": "text",
                "text": metadata_text,
            }
        ]

        # Interleave filename string labels immediately preceding their raw image payloads
        for filename, image_data in zip(self._image_filenames, self._images):
            filename_payload = {
                "type": "text",
                "text": f"\n--- VISUAL EVIDENCE WORKSPACE FOR IDENTIFIER: {filename} ---",
            }

            image_payload = {
                "type": "image_url",
                "image_url": {"url": image_data, "detail": "high"},
            }

            user_content_payload.append(filename_payload)
            user_content_payload.append(image_payload)

        # Format the targeted operational engine profile
        developer_content = DEVELOPER_PROMPT_INSTRUCTIONS.format(model_name=MODEL_NAME)

        return [
            {"role": "developer", "content": developer_content},
            {"role": "user", "content": user_content_payload},
        ]

    def get_response(self) -> dict:
        if not self._client:
            return {}

        message_attrs = self._get_message()

        response = self._client.chat.completions.create(
            model=MODEL_NAME,
            messages=message_attrs,
            max_completion_tokens=MAX_TOKENS,
            response_format=self._get_response_format(),
            temperature=0.0,
            top_p=1.0,
            timeout=90.0,
        )

        # Pluck prompt aka user_content_payload from message attribute to
        # track usage.
        prompt = message_attrs[1].get("content") or ""

        return self._filter_response(response, prompt)

    def _filter_response(self, response: ChatCompletion, prompt: str) -> dict:
        # Extrapolate response
        # model: str, input_tokens: int, output_tokens: int, prompt: str
        usage = getattr(response, "usage", None)
        if usage is not None:
            self.usage.record(
                model=response.model,
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                prompt=prompt,
                response_id=response.id,
            )

        try:
            content_str = response.choices[0].message.content
            if not content_str:
                return {}

            content = json.loads(content_str)

        except (AttributeError, IndexError, json.JSONDecodeError) as e:
            print(f"❌ Error occurred while parsing response: {e}")
            return {}

        if usage is not None:
            content["usage"] = usage.model_dump()

        return content

    def _get_response_format(self) -> dict:
        """
        https://developers.openai.com/api/reference/python/resources/chat/subresources/completions/methods/create

        An object specifying the format that the model must output.
        Setting to { "type": "json_schema", "json_schema": {...} } enables
        Structured Outputs which ensures the model
        will match your supplied JSON schema. Learn more in the Structured
        Outputs guide.

        Setting to { "type": "json_object" } enables the older JSON mode, which
        ensures the message the model generates is valid JSON. Using
        json_schema is preferred for models that support it.
        :return:
        """
        return {
            "type": "json_schema",
            "json_schema": self._get_json_schema(),
        }

    def _get_json_schema(self) -> dict:
        """
        https://developers.openai.com/api/docs/guides/structured-outputs
        Ensure text responses from the model adhere to a JSON schema you define.
        :return:
        """

        return {
            "name": self.name.replace(" ", "_").lower(),
            "description": "Fetches claim analysis for each dataset row from a csv file and returns the results.",
            "strict": True,
            "schema": RESPONSE_SCHEMA,
        }

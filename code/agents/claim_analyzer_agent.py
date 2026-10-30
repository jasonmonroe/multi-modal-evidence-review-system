# code/agents/claim_analyzer_agent.py

# +---------------------------------------------------------------------------+
# |                              CLAIM ANALYZER                               |
# +---------------------------------------------------------------------------+

# Python Libraries
import pandas as pd

# Local Libraries
from models.insurance_agent_model import InsuranceAgentModel
from src.constants import DECISION_C, DECISION_N, MODEL_NAME
from src.data_handler import DataHandler
from src.image_handler import ImageHandler
from src.utils import log_chat_transcript, show_banner, show_timer, start_timer


class ClaimAnalyzer:
    def __init__(self, data: DataHandler, image: ImageHandler):
        self.data = data
        self.image = image

        self._model = InsuranceAgentModel(len(data.claims))
        self._row = pd.Series()

    def get_by_index(self, index: int) -> dict:
        """
        Extract the entire row as a Series
        :param index:
        :return:
        """

        # Pull the row by index and then encode the images
        self._row = self.data.claims.iloc[index]
        image_paths = self._row["image_paths"].split(";")
        user_id = self._row["user_id"]

        self.image.load(image_paths)

        # Parse fields from the row
        return {
            "user_id": user_id,
            "image_paths": self._row["image_paths"],
            "user_claim": self._format_user_claim(self._row["user_claim"]),
            "claim_object": self._row["claim_object"],
            "images": self.image.images,
            "image_filenames": self.image.filenames,
            "user_history": self._get_user_history(user_id),  # dataframe
            # Note: 💡 Include row for reference
            "csv_row": self._row,
        }

    def _format_user_claim(self, user_claim: str) -> str:
        """
        Clean and readable transcript for the LLM.
        :param user_claim:
        :return: string
        """

        return user_claim.replace(" | ", "\n").strip()

    def _get_user_history(self, user_id: str) -> pd.DataFrame:
        """
        Filters history by user id
        :param user_id:
        :return:
        """
        history_df = self.data.user_history
        return history_df[history_df["user_id"] == user_id]

    def _generate_evidence_prompt(self) -> str:
        """
        Generates a prompt for evidence that has the rules and then converts
        the instructions to a string.
        :return:
        """

        # Generate a prompt pertaining to the evidence policy.  Include
        # instructions for the LLM to understand it.
        instructions = [
            "You must enforce the following strict evidence validation rules per company policy.",
            "If any rule fails, immediately set `evidence_standard_met` to `false`.",
        ]

        claim_object = self._row["claim_object"]
        requirements_df = self.data.evidence_reqs

        # Separate all rules and claim object specific rules and then combine
        # them for relevant usage.
        all_requirements_df = requirements_df["claim_object"] == "all"
        claim_object_requirements_df = requirements_df["claim_object"] == claim_object

        # Combine rules that apply to 'all' objects and those specific to the
        # current claim_object.
        # We include all rules for the object because the specific issue_type
        # isn't known until the LLM analyzes the images.
        requirements = requirements_df[
            all_requirements_df | claim_object_requirements_df
        ]

        for row_df in requirements.itertuples():
            rule_no = len(instructions)
            req_id = row_df.requirement_id
            applies_to = row_df.applies_to
            min_img_evidence = row_df.minimum_image_evidence

            instructions.append(
                f"{rule_no}. [RULE: {req_id}] for claims involving '{applies_to}':\n"
                f" - MANDATE: {min_img_evidence}\n"
                f" - FAILURE CONDITION: If the submitted visual evidence is blurry, obscured, missing from the frame, "
                f"or taken from an unreadable distance/angle, this requirement is NOT satisfied.".strip()
            )

        # Return evidence prompt as a string
        return "\n\n".join(instructions).strip()

    def show_response(self, response: dict) -> None:
        subtitles = []
        for key, value in response.items():
            subtitles.append(f"ℹ️ {key.title().replace('_', ' ')}: {value}")

        show_banner(f"Model: {MODEL_NAME} (Response)".upper(), subtitles)

    def call(self, dataset: dict) -> str | dict | None:
        """
        Calls the InsuranceAgentModel to get a response based on the dataset
        provided.
        dataset must be set.
        :param dataset:
        :return:
        """

        if dataset:
            self._model.update(dataset)

        start_time = start_timer()
        response = self._model.get_response()
        show_timer(start_time)

        if isinstance(response, dict):
            self.show_response(response)

        return response

    def analyze_claim(self, index: int, strategy: str) -> dict:
        """
        The Task: Playing Detective
        Access specific columns using the column header names

        :param strategy:
        :param index:
        :return:
        """

        # Get row of data as a baseline
        dataset = self.get_by_index(index)
        dataset["evidence"] = self._generate_evidence_prompt()
        dataset["strategy"] = strategy

        # LOG INPUT TRANSCRIPT: Log the exact context & rules being sent
        input_payload_summary = (
            f"User ID: {dataset['user_id']}\n"
            f"Claim Object: {dataset['claim_object']}\n"
            f"Strategy Config: {strategy}\n\n"
            f"--- GENERATED EVIDENCE RULES PROMPT ---\n"
            f"{dataset['evidence']}\n\n"
            f"--- CLEANED USER CLAIM TRANSCRIPT ---\n"
            f"{dataset['user_claim']}"
        )
        log_chat_transcript(f"ROW INDEX {index} - INPUT PAYLOAD", input_payload_summary)

        # Send dataset to the insurance agent model for analysis
        response = self.call(dataset)

        # Merge response with investigative dataset
        dataset = {**dataset, **response}

        # Log Model Response: Log exactly what the raw LLM outputted
        log_chat_transcript(f"ROW INDEX {index} - RAW LLM RESPONSE", str(response))

        # Although the LLM provides a claim status and risk flags, we need to
        # apply deterministic business logic to ensure consistency and
        # compliance with company policy. Review analysis, make a final decision
        # based on user history and risk assessment.
        dataset = self._make_decision(dataset)

        # LOG FINAL RESOLUTION: Record the deterministic final output state
        final_decision_summary = (
            f"Resolved Risk Flags: {dataset['risk_flags']}\n"
            f"Overridden Claim Status: {dataset['claim_status']}"
        )

        log_chat_transcript(
            f"ROW INDEX {index} - POST-PROCESS MAKE DECISION", final_decision_summary
        )

        return dataset

    def _make_decision(self, dataset: dict) -> dict:
        """
        Take the risk_flags and claim status from the LLM and make a final
        decision based on user history and risk assessment.

        :param dataset:
        :return:
        """

        user_history = dataset.get("user_history")
        user_history_has_risk = (
            "user_history_risk" in user_history["history_flags"].values
        )

        # Parse the LLM's suggested fields
        risk_flags = set(dataset.get("risk_flags", "none").split(";"))
        claim_status = dataset.get("claim_status")
        valid_image = dataset.get("valid_image", True)
        severity = dataset.get("severity", "unknown")
        evidence_standard_met = dataset.get("evidence_standard_met", True)

        # Clean the "none" string from set operations
        risk_flags.discard("none")

        # Override 1 & 3: Inject historical risk or prompt injection review
        if user_history_has_risk:
            risk_flags.add("user_history_risk")
            risk_flags.add("manual_review_required")

        if (
            "text_instruction_present" in risk_flags
            or "possible_manipulation" in risk_flags
        ):
            claim_status = DECISION_C
            risk_flags.add("manual_review_required")
            valid_image = False

        # Override 2: Stock photos / screenshots
        if "non_original_image" in risk_flags:
            valid_image = False

        # Override 6: Minimum evidence standards
        if not evidence_standard_met:
            claim_status = DECISION_N

        # Override 5: Severity safety valve
        if claim_status == DECISION_N and severity not in ["none", "unknown"]:
            severity = "unknown"

        # Format risk flags back to expected CSV format
        final_risk_flags = ";".join(sorted(risk_flags)) if risk_flags else "none"

        new_dataset = {
            "claim_status": claim_status,
            "evidence_standard_met": evidence_standard_met,
            "risk_flags": final_risk_flags,
            "severity": severity,
            "valid_image": valid_image,
        }

        dataset.update(new_dataset)

        return dataset

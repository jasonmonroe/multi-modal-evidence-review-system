# code/src/claim_handler.py

# +---------------------------------------------------------------------------+
# |                               CLAIM HANDLER                               |
# +---------------------------------------------------------------------------+

# Python Libraries
import csv

# Local Libraries
from src.constants import TEST_FILE


class ClaimHandler:
    """
    Claim Handler used for output
    """

    def __init__(self):
        """
        Columns for output.csv
        """
        self.user_id = None
        self.image_paths = None
        self.user_claim = None
        self.claim_object = None
        self.evidence_standard_met = None
        self.evidence_standard_met_reason = None
        self.risk_flags = None
        self.issue_type = None
        self.object_part = None
        self.claim_status = None
        self.claim_status_justification = None
        self.supporting_image_ids = None
        self.valid_image = None
        self.severity = None

    def update(self, dataset: dict) -> None:
        """
        Update attributes with new dataset values.
        :param dataset:
        :return:
        """
        for attr_name, value in dataset.items():
            if hasattr(self, attr_name):
                setattr(self, attr_name, value)

    def _format_values_for_output(self) -> None:
        # Format values for output
        if "\n" in self.user_claim:
            self.user_claim = self.user_claim.replace("\n", " | ").strip()

        if isinstance(self.image_paths, list):
            self.image_paths = ";".join(self.image_paths).strip()

        if isinstance(self.supporting_image_ids, list):
            self.supporting_image_ids = ";".join(
                str(id) for id in self.supporting_image_ids
            )
            self.supporting_image_ids = self.supporting_image_ids.strip()

        if isinstance(self.evidence_standard_met, bool):
            self.evidence_standard_met = str(self.evidence_standard_met).lower()

        if isinstance(self.evidence_standard_met_reason, str):
            self.evidence_standard_met_reason = (
                self.evidence_standard_met_reason.strip()
            )

        if isinstance(self.claim_status_justification, str):
            self.claim_status_justification = self.claim_status_justification.strip()

        if isinstance(self.valid_image, bool):
            self.valid_image = str(self.valid_image).lower()

    def export_csv_row(self) -> list:
        # Format values for output (handles your quotes, etc.)
        self._format_values_for_output()

        csv_row = []
        for value in self.__dict__.values():
            if isinstance(value, str):
                # Clean up apostrophes from strings safely
                value = value.replace("'", "").replace("’", "")

            # Convert None to empty spaces seamlessly
            elif value is None:
                value = ""

            csv_row.append(value)

        return csv_row

    def progress(self, idx: int, total: int) -> str:
        """
        Displays progress of claim analysis.

        :param idx:
        :param total:
        :return:
        """
        print("\n")

        i_empty, i_full = "☑️", "✅️"
        completion_pct = ((idx + 1) / total) * 100

        graphic = ""
        for i in range(total):
            graphic += i_full if i <= idx else i_empty

        return graphic + f"\t{completion_pct:.1f}%"

    def write_output(self, csv_rows: list) -> None:
        """
        Write the output to a CSV file.
        Use writerows to cleanly unpack a list of multiple rows!
        :param csv_rows:
        :return:
        """
        print(f"\n# --- ✏️ Writing output to {TEST_FILE} --- #\n")

        with open(TEST_FILE, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerows(csv_rows)

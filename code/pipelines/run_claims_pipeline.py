# code/pipelines/run_claims_pipeline.py

# +---------------------------------------------------------------------------+
# |                              CLAIMS PIPELINE                              |
# +---------------------------------------------------------------------------+

# Python Libraries
from time import sleep

import openai

# Local Libraries
from evaluation.claim_analyzer import ClaimAnalyzer
from evaluation.claim_handler import ClaimHandler
from src.constants import (
    CSV_HEADERS,
    PAUSE_TIMER,
    SLEEP_TIMER,
    SLEEP_TIMER_INC,
)
from src.data_handler import DataHandler
from src.image_handler import ImageHandler


def run_claims_pipeline(data_handle: DataHandler, image_handle: ImageHandler) -> None:
    print("\n--- 🏃 Running Claims Pipeline 🏃 ---")

    # Extract the actual damage claim from the conversation
    analyzer = ClaimAnalyzer(data_handle, image_handle)
    claim = ClaimHandler()

    # For each row in `claims.csv`, generate one row in `output.csv`
    csv_rows = [list(CSV_HEADERS)]
    csv_row_cnt = len(analyzer.data.claims)

    # Analyze each claim from the dataset case by case
    extra_time = 0
    for idx, row in enumerate(analyzer.data.claims.itertuples()):
        print(f"\n# --- Claim analysis for row {idx + 1} of {csv_row_cnt} begins --- #")

        while True:
            try:
                # Analyzing claim and get final dataset.
                final_dataset = analyzer.analyze_claim(idx, "Basic Prompt")
                break  # Success! Break the retry loop and move forward

            except openai.RateLimitError as e:
                print(f"{e}")
                print(
                    f"\n⚠️ Hit Token Quota Limit! Pausing pipeline for {SLEEP_TIMER} seconds for another attempt..."
                )
                sleep(SLEEP_TIMER)
                extra_time += SLEEP_TIMER_INC
                print(f"🔄 Adding extra time ({extra_time}).  Resuming analysis...")

        # Create CSV data
        claim.update(final_dataset)
        csv_rows.append(claim.export_csv_row())

        # Consistent pacing between normal loops
        print(claim.progress(idx, csv_row_cnt))
        sleep(PAUSE_TIMER + extra_time)

    # Write the output to a file
    claim.write_output(csv_rows)

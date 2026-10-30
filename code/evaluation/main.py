# code/evaluation/main.py

# +---------------------------------------------------------------------------+
# |                            EVALUATION PIPELINE                            |
# +---------------------------------------------------------------------------+


from agents.claim_analyzer_agent import ClaimAnalyzer
from src.constants import (
    MODEL_NAME,
    PAUSE_TIMER,
    REPORT_FILE,
    SAMPLE_CLAIMS_FILE,
    SECS_IN_MIN,
    SLEEP_TIMER,
    SLEEP_TIMER_INC,
    TOKEN_UNIT,
)
from src.data_handler import DataHandler
from src.image_handler import ImageHandler
from src.utils import show_banner, start_timer


def run_evaluation_report_pipeline(
    data_handle: DataHandler, image_handle: ImageHandler
) -> None:
    """
    You write a script (usually in code/evaluation/main.py) that loops through sample_claims.csv, compares your model's
    predictions to the expected_output columns, and calculates your accuracy. This script should also track how many
    tokens you use and how long each call takes.

    :param data_handle:
    :param image_handle:
    :return:
    """

    print("\n--- 🏃 Running Evaluation Report Pipeline 🏃 ---")
    print("# --- Evaluation Workflow Start --- #")

    show_banner(
        "Evaluation Report".upper(),
        [
            f"Evaluating model performance on {SAMPLE_CLAIMS_FILE}.",
            "Evaluating Strategies: `Basic` and `Chain-of-Thought` Prompts",
        ],
        center_subtitle_text=True,
    )

    # Explicitly load the evaluation answer key sample dataset
    data_handle.load_data({"sample": True})
    sample_data = data_handle.claims
    sample_data_len = len(sample_data)

    results = {}
    strategies = ["Basic Prompt", "Chain-of-Thought Prompt"]

    for strategy in strategies:
        analyzer = ClaimAnalyzer(data_handle, image_handle)

        metrics = {
            "correct": 0,
            "total": sample_data_len,
            "total_tokens": 0,
            "total_cost": 0.0,
            "start_time": start_timer(),
        }

        for idx in range(sample_data_len):
            # Pass the dynamic loop strategy directly to your analyzer!
            print(
                f"\n# --- Evaluating {strategy} - Row {idx + 1} of {sample_data_len} --- #"
            )

            response = analyzer.analyze_claim(idx, strategy=strategy)

            # Ensure 'expected_claim_status' is your actual CSV ground truth column
            ground_truth = sample_data.iloc[idx].get(
                "expected_claim_status", sample_data.iloc[idx]["claim_status"]
            )

            if response["claim_status"] == ground_truth:
                metrics["correct"] += 1

            # Metric accounting remains standard
            metrics["total_tokens"] += response.get("usage", {}).get("total_tokens", 0)
            metrics["total_cost"] = (metrics["total_tokens"] / TOKEN_UNIT) * 0.01

        metrics["duration"] = start_timer() - metrics["start_time"]
        metrics["accuracy"] = (
            (metrics["correct"] / metrics["total"]) * 100 if metrics["total"] > 0 else 0
        )
        results[strategy] = metrics

    subtitles = []
    for strategy, metric in results.items():
        subtitles.extend(f"Strategy: {strategy}")
        subtitles.extend(f"🎯️Accuracy: {metric['accuracy']:.2f}%")
        subtitles.extend(f"ℹ️Total Time: {metric['duration']:.2f}s")
        subtitles.extend(f"ℹ️Tokens: {metric['total_tokens']}")
        subtitles.extend(f"💰️Est Cost: ${metric['total_cost']:.2f}\n")

    show_banner("Evaluation Report".upper(), subtitles)

    print("# --- Evaluation Workflow Complete --- #")

    # Now it's time to generate a report template to be used.
    generate_report_template(results)


def _pick_best_strategy(results: dict) -> tuple[str, str]:
    """
    Choose the strategy to recommend: highest accuracy, then lowest cost, then
    shortest runtime. Returns the strategy name and a one-sentence reason.
    """

    def rank(item: tuple) -> tuple:
        _, m = item
        return m["accuracy"], -m["total_cost"], -m["duration"]

    name, best = max(results.items(), key=rank)
    others = [m for n, m in results.items() if n != name]

    if not others:
        return name, f"it was the only strategy evaluated ({best['accuracy']:.2f}% accuracy)"

    if all(o["accuracy"] == best["accuracy"] for o in others):
        if all(o["total_cost"] == best["total_cost"] for o in others):
            return name, (
                f"accuracy ({best['accuracy']:.2f}%) and cost were tied, and it "
                f"had the shortest runtime ({best['duration']:.2f}s)"
            )
        return name, (
            f"accuracy was tied at {best['accuracy']:.2f}%, and it had the "
            f"lowest estimated cost (${best['total_cost']:.2f})"
        )

    return name, f"it had the highest accuracy ({best['accuracy']:.2f}%)"


def generate_report_template(results: dict) -> None:
    """
    Generates the operational evaluation report and writes it to REPORT_FILE
    as Markdown.

    :param results: strategy name -> metrics dict, as built by
        run_evaluation_report_pipeline (accuracy, correct, total, duration,
        total_tokens, total_cost).
    :return:
    """

    print("# --- 📑 Generating Report 📑 --- #")

    strategy_rows = []
    sample_count = 0
    calls = 0
    total_tokens_sum = 0
    total_cost_sum = 0.0
    total_duration = 0.0

    for strategy, metrics in results.items():
        total = metrics["total"]
        avg_latency = metrics["duration"] / total if total else 0.0

        strategy_rows.append(
            f"| {strategy} | {total} | {metrics['correct']} "
            f"| {metrics['accuracy']:.2f}% | {metrics['duration']:.2f}s "
            f"| {avg_latency:.2f}s | {metrics['total_tokens']:,} "
            f"| ${metrics['total_cost']:.2f} |"
        )
        sample_count = max(sample_count, total)
        calls += total
        total_tokens_sum += metrics["total_tokens"]
        total_cost_sum += metrics["total_cost"]
        total_duration += metrics["duration"]

    minutes = total_duration / SECS_IN_MIN
    rpm = calls / minutes if minutes else 0.0
    tpm = total_tokens_sum / minutes if minutes else 0.0

    if results:
        best, reason = _pick_best_strategy(results)
        choice = f"We chose **{best}** because {reason}."
    else:
        choice = "No strategies were evaluated."

    lines = [
        "# Operational Evaluation Report",
        "",
        f"Evaluated on `{SAMPLE_CLAIMS_FILE}` ({sample_count} labeled claims).",
        "Accuracy is the share of rows where the predicted `claim_status` matches the label.",
        "",
        "### Strategy Comparison",
        "| Strategy | Samples | Correct | Accuracy | Total Time | Avg Latency / Claim | Tokens | Cost (est) |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        *strategy_rows,
        "",
        "## Operational Analysis",
        f"- **Model Used:** {MODEL_NAME}",
        f"- **Model Calls:** {calls}",
        f"- **Total Token Usage:** {total_tokens_sum:,}",
        f"- **Approximate Cost:** ${total_cost_sum:.2f} "
        f"(assumes a flat $0.01 per {TOKEN_UNIT:,} tokens)",
        f"- **Throughput:** about {rpm:.1f} requests/min and {tpm:,.0f} tokens/min",
        f"- **Pacing and Retries:** {PAUSE_TIMER}s pause between claims; on a "
        f"rate-limit error the pipeline waits {SLEEP_TIMER}s and retries, "
        f"adding {SLEEP_TIMER_INC}s of extra pacing per retry",
        "",
        "## Final Strategy Choice",
        choice,
        "",
    ]

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

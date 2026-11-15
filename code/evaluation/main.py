# code/evaluation/main.py

# +---------------------------------------------------------------------------+
# |                            EVALUATION PIPELINE                            |
# +---------------------------------------------------------------------------+


import os

from agents.claim_analyzer_agent import ClaimAnalyzer
from src.constants import (
    DECISION_C,
    DECISION_N,
    DECISION_S,
    MODEL_NAME,
    PAUSE_TIMER,
    REPORT_FILE,
    SAMPLE_CLAIMS_FILE,
    SECS_IN_MIN,
    SLEEP_TIMER,
    SLEEP_TIMER_INC,
    INPUT_PRICE_PER_UNIT,
    OUTPUT_PRICE_PER_UNIT,
    PRICING_MODEL,
    TOKEN_UNIT,
)
from src.data_handler import DataHandler
from src.image_handler import ImageHandler
from src.utils import show_banner, start_timer


def run_evaluation_report_pipeline(
    data_handle: DataHandler, image_handle: ImageHandler
) -> None:
    """
    Run every prompting strategy over the labeled sample set and compare each
    predicted `claim_status` with the label. Tracks accuracy, a confusion
    matrix, tokens, estimated cost and runtime per strategy, then writes the
    evaluation report.

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

    label_column = (
        "expected_claim_status"
        if "expected_claim_status" in sample_data.columns
        else "claim_status"
    )
    baseline = _majority_baseline(sample_data, label_column)

    results = {}
    strategies = ["Basic Prompt", "Chain-of-Thought Prompt"]

    for strategy in strategies:
        analyzer = ClaimAnalyzer(data_handle, image_handle)

        metrics = {
            "correct": 0,
            "total": sample_data_len,
            "total_tokens": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_cost": 0.0,
            "start_time": start_timer(),
            "confusion": {},
        }

        for idx in range(sample_data_len):
            # Pass the dynamic loop strategy directly to your analyzer!
            print(
                f"\n# --- Evaluating {strategy} - Row {idx + 1} of {sample_data_len} --- #"
            )

            response = analyzer.analyze_claim(idx, strategy=strategy)

            ground_truth = sample_data.iloc[idx][label_column]

            predicted = response["claim_status"]
            if predicted == ground_truth:
                metrics["correct"] += 1

            row = metrics["confusion"].setdefault(ground_truth, {})
            row[predicted] = row.get(predicted, 0) + 1

            # Metric accounting remains standard
            usage = response.get("usage", {})
            metrics["total_tokens"] += usage.get("total_tokens", 0)
            metrics["input_tokens"] += usage.get("prompt_tokens", 0)
            metrics["output_tokens"] += usage.get("completion_tokens", 0)
            metrics["total_cost"] = _token_cost(
                metrics["input_tokens"], metrics["output_tokens"]
            )

        metrics["duration"] = start_timer() - metrics["start_time"]
        metrics["accuracy"] = (
            (metrics["correct"] / metrics["total"]) * 100 if metrics["total"] > 0 else 0
        )
        results[strategy] = metrics

    subtitles = []
    for strategy, metric in results.items():
        subtitles.append(f"Strategy: {strategy}")
        subtitles.append(f"🎯️Accuracy: {metric['accuracy']:.2f}%")
        subtitles.append(f"ℹ️Total Time: {metric['duration']:.2f}s")
        subtitles.append(f"ℹ️Tokens: {metric['total_tokens']}")
        subtitles.append(f"💰️Est Cost: {_format_cost(*_cost_range(metric))}")

    show_banner("Evaluation Report".upper(), subtitles)

    print("# --- Evaluation Workflow Complete --- #")

    # Now it's time to generate a report template to be used.
    generate_report_template(results, baseline=baseline)


def _token_cost(input_tokens: int, output_tokens: int) -> float:
    """Dollar cost of a call, pricing input and output tokens separately."""
    return (
        input_tokens * INPUT_PRICE_PER_UNIT + output_tokens * OUTPUT_PRICE_PER_UNIT
    ) / TOKEN_UNIT


def _cost_range(metrics: dict) -> tuple[float, float]:
    """
    (low, high) cost for one strategy. Exact when the input/output split was
    recorded; otherwise bounded by pricing every token as input (low) or as
    output (high), because the split is unknown.
    """
    if "input_tokens" in metrics and "output_tokens" in metrics:
        cost = _token_cost(metrics["input_tokens"], metrics["output_tokens"])
        return cost, cost

    total = metrics["total_tokens"]
    return _token_cost(total, 0), _token_cost(0, total)


def _format_cost(low: float, high: float) -> str:
    if round(low, 2) == round(high, 2):
        return f"${low:.2f}"
    return f"${low:.2f} to ${high:.2f}"


def _majority_baseline(sample_data, label_column: str) -> dict | None:
    """
    Accuracy of a classifier that always predicts the most common label.
    Returns None for an empty or unlabeled sample set.
    """

    if label_column not in sample_data.columns or sample_data.empty:
        return None

    counts = sample_data[label_column].value_counts()
    return {
        "label": counts.index[0],
        "count": int(counts.iloc[0]),
        "total": len(sample_data),
        "accuracy": counts.iloc[0] / len(sample_data) * 100,
    }


def _pick_best_strategy(results: dict) -> tuple[str, str]:
    """
    Choose the strategy to recommend: highest accuracy, then lowest cost, then
    shortest runtime. Returns the strategy name and a one-sentence reason.
    """

    def cost(m: dict) -> float:
        return sum(_cost_range(m)) / 2

    def rank(item: tuple) -> tuple:
        _, m = item
        return m["accuracy"], -cost(m), -m["duration"]

    name, best = max(results.items(), key=rank)
    others = [m for n, m in results.items() if n != name]

    if not others:
        return name, f"it was the only strategy evaluated ({best['accuracy']:.2f}% accuracy)"

    if all(o["accuracy"] == best["accuracy"] for o in others):
        if all(cost(o) == cost(best) for o in others):
            return name, (
                f"accuracy ({best['accuracy']:.2f}%) and cost were tied, and it "
                f"had the shortest runtime ({best['duration']:.2f}s)"
            )
        return name, (
            f"accuracy was tied at {best['accuracy']:.2f}%, and it had the "
            f"lowest estimated cost ({_format_cost(*_cost_range(best))})"
        )

    return name, f"it had the highest accuracy ({best['accuracy']:.2f}%)"


def _baseline_lines(results: dict, baseline: dict | None) -> list[str]:
    """Markdown section comparing the best strategy with the majority baseline."""

    if not baseline:
        return []

    best_accuracy = max((m["accuracy"] for m in results.values()), default=0.0)
    gap = best_accuracy - baseline["accuracy"]

    if gap > 0:
        verdict = f"The best strategy beats it by {gap:.2f} percentage points."
    elif gap == 0:
        verdict = "The best strategy only matches it."
    else:
        verdict = (
            f"The best strategy scores {best_accuracy:.2f}%, which is "
            f"{abs(gap):.2f} points below it."
        )

    return [
        "## Baseline",
        f"A majority-class baseline that always answers `{baseline['label']}` "
        f"scores {baseline['accuracy']:.2f}% ({baseline['count']} of "
        f"{baseline['total']} claims). {verdict}",
        "",
    ]


def _confusion_lines(strategy: str, confusion: dict) -> list[str]:
    """Markdown confusion matrix: actual label per row, predicted per column."""

    labels = [DECISION_S, DECISION_C, DECISION_N]
    seen = set(confusion) | {p for row in confusion.values() for p in row}
    labels += sorted(str(l) for l in seen - set(labels))

    lines = [
        f"### Confusion Matrix: {strategy}",
        "| Actual \\ Predicted | " + " | ".join(labels) + " |",
        "| :--- | " + " | ".join("---:" for _ in labels) + " |",
    ]
    for actual in labels:
        row = confusion.get(actual, {})
        lines.append(
            f"| {actual} | " + " | ".join(str(row.get(p, 0)) for p in labels) + " |"
        )
    lines.append("")
    return lines


def generate_report_template(results: dict, baseline: dict | None = None) -> None:
    """
    Generates the operational evaluation report and writes it to REPORT_FILE
    as Markdown.

    :param results: strategy name -> metrics dict, as built by
        run_evaluation_report_pipeline (accuracy, correct, total, duration,
        total_tokens, total_cost and optionally confusion).
    :param baseline: majority-class baseline from _majority_baseline, if known.
    :return:
    """

    print("# --- 📑 Generating Report 📑 --- #")

    strategy_rows = []
    sample_count = 0
    calls = 0
    total_tokens_sum = 0
    cost_low = cost_high = 0.0
    input_sum = output_sum = 0
    split_known = True
    total_duration = 0.0

    for strategy, metrics in results.items():
        total = metrics["total"]
        avg_latency = metrics["duration"] / total if total else 0.0

        low, high = _cost_range(metrics)
        strategy_rows.append(
            f"| {strategy} | {metrics['correct']} / {total} "
            f"| {metrics['accuracy']:.2f}% | {metrics['duration']:.2f}s "
            f"| {avg_latency:.2f}s | {_format_cost(low, high)} |"
        )
        sample_count = max(sample_count, total)
        calls += total
        total_tokens_sum += metrics["total_tokens"]
        cost_low += low
        cost_high += high
        if "input_tokens" in metrics and "output_tokens" in metrics:
            input_sum += metrics["input_tokens"]
            output_sum += metrics["output_tokens"]
        else:
            split_known = False
        total_duration += metrics["duration"]

    minutes = total_duration / SECS_IN_MIN
    rpm = calls / minutes if minutes else 0.0
    tpm = total_tokens_sum / minutes if minutes else 0.0

    if results:
        best, reason = _pick_best_strategy(results)
        choice = f"We chose **{best}** because {reason}."
    else:
        choice = "No strategies were evaluated."

    confusion_lines = []
    for strategy, metrics in results.items():
        if metrics.get("confusion"):
            confusion_lines += _confusion_lines(strategy, metrics["confusion"])

    lines = [
        "# Operational Evaluation Report",
        "",
        f"Evaluated on `{os.path.basename(SAMPLE_CLAIMS_FILE)}` ({sample_count} labeled claims).",
        "Accuracy is the share of rows where the predicted `claim_status` matches the label.",
        "",
        "## Strategy Comparison",
        "| Strategy | Correct | Accuracy | Total Time | Avg Latency / Claim | Cost (est) |",
        "| :--- | ---: | ---: | ---: | ---: | ---: |",
        *strategy_rows,
        "",
        *_baseline_lines(results, baseline),
        *confusion_lines,
        "## Operational Analysis",
        f"- **Model Used:** {MODEL_NAME}",
        f"- **Model Calls:** {calls}",
        f"- **Total Token Usage:** {total_tokens_sum:,}"
        + (
            f" ({input_sum:,} input, {output_sum:,} output)"
            if split_known and results
            else ""
        ),
        f"- **Approximate Cost:** {_format_cost(cost_low, cost_high)}",
        f"- **Pricing Assumption:** {PRICING_MODEL} at "
        f"${INPUT_PRICE_PER_UNIT:.2f} per 1M input tokens and "
        f"${OUTPUT_PRICE_PER_UNIT:.2f} per 1M output tokens, as reported by "
        "third-party price trackers (Google's pricing page does not currently "
        "list this model; verify before budgeting)."
        + (
            ""
            if split_known
            else " This run did not record the input/output token split, so "
            "cost is shown as a range: all tokens priced as input (low) to "
            "all priced as output (high)."
        ),
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

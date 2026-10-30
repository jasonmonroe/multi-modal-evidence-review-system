# Multi-Modal Evidence Review System

An automated reviewer for insurance-style damage claims. It looks at the photos a customer submitted, reads the claim
conversation, and decides whether the evidence backs the claim up.

> Post-hackathon rework of my June 2026 hackathon entry. The original, as submitted, is preserved at the git tag
> `hackathon-submission-june26`.

The system verifies damage claims for three object types — `car`, `laptop`, and `package` — by combining:

- one or more submitted **images** (the primary source of truth)
- a short **claim conversation** (defines what needs to be checked)
- the user's **claim history** (risk context only; never overrides clear visual evidence)
- the **minimum evidence requirements** for the object and issue type

For each claim it decides whether the images **support** the claim, **contradict** it, or provide **not enough
information**, and emits a structured row of predictions (issue type, object part, severity, risk flags, supporting
image IDs, and short image-grounded justifications).

The images are the primary source of truth. The claim conversation defines what needs to be checked, and user history
only adds risk context; it never overrides clear visual evidence on its own.

The model's answer is not final. A deterministic rule layer (`ClaimAnalyzer._make_decision`) then applies company
policy: it injects history risk flags, forces `contradicted` for prompt-injection or manipulation, invalidates
non-original images, and downgrades to `not_enough_information` when the evidence standard is not met.

The original task brief, including the full I/O schema and allowed values, is kept in
[problem_statement.md](problem_statement.md) for reference.

---

## Table of Contents

1. [How It Works](#how-it-works)
2. [Project Structure](#project-structure)
3. [Data File Structure](#data-file-structure)
4. [Installation](#installation)
5. [Running the System](#running-the-system)
6. [Expected Output](#expected-output)
7. [Running the Evaluation](#running-the-evaluation)
8. [Results](#results)
9. [Running the Unit Tests](#running-the-unit-tests)
10. [Input and Output Schema](#input-and-output-schema)
11. [Cost, Latency, and Rate Limits](#cost-latency-and-rate-limits)
12. [Troubleshooting](#troubleshooting)
13. [Limitations and Next Steps](#limitations-and-next-steps)

---

## How It Works

1. `DataHandler` loads `claims.csv` (or `sample_claims.csv` with `--sample`), `user_history.csv`, and
   `evidence_requirements.csv`.
2. `ImageHandler` reads the images referenced by each claim row and base64-encodes them as data URLs.
3. `ClaimAnalyzer` builds a prompt (claim conversation + images + matching evidence requirements) and calls a
   vision-capable LLM through an OpenAI-compatible API (`InsuranceAgentModel`), requesting a JSON response that
   conforms to `RESPONSE_SCHEMA`.
4. `ClaimAnalyzer._make_decision` applies the deterministic overrides described above, using the user's history.
5. `ClaimHandler` normalizes the result (lists joined with `;`, booleans lowercased, apostrophes removed) and exports
   one CSV row per claim.
6. The pipeline retries on rate-limit errors with a back-off pause and paces requests between calls.
7. The evaluation pipeline runs the same analyzer over `sample_claims.csv`, compares predictions with the expected
   labels, and tracks accuracy, token usage, cost, and timing for each prompting strategy (`Basic Prompt` and
   `Chain-of-Thought Prompt`).

---

## Project Structure

```text
.
├── AGENTS.md                         # Agent/contributor rules and project contract
├── CLAUDE.md                         # Points Claude Code at AGENTS.md
├── README.md                         # You are here
├── problem_statement.md              # Original task brief and I/O schema
├── requirements.txt                  # Python dependencies
├── .env.example                      # Template for required environment variables
├── code/                             # The solution (run all commands from here)
│   ├── main.py                       # Terminal entry point (claims pipeline + optional evaluation)
│   ├── agents/
│   │   └── claim_analyzer_agent.py   # ClaimAnalyzer: prompts the model and parses its response
│   ├── models/
│   │   └── insurance_agent_model.py  # Model / schema definitions for the insurance agent
│   ├── pipelines/
│   │   └── run_claims_pipeline.py    # Runs claims.csv -> output.csv with retry and pacing
│   ├── src/
│   │   ├── constants.py              # Paths, allowed values, CSV headers, JSON response schema
│   │   ├── claim_handler.py          # Normalizes a model result into an output row
│   │   ├── data_handler.py           # Loads claims, user history, and evidence requirements
│   │   ├── image_handler.py          # Loads and prepares images
│   │   └── utils.py                  # Banner, timer, and helper utilities
│   ├── evaluation/
│   │   ├── main.py                   # Evaluation entry point (sample_claims.csv)
│   │   ├── usage_tracker.py          # Token / cost / latency tracking
│   │   └── evaluation_report.md      # Operational analysis (calls, tokens, cost, latency)
│   ├── pytest.ini                    # pytest configuration
│   ├── .coveragerc                   # Coverage configuration
│   └── tests/                        # Unit tests (mirrors agents/, evaluation/, models/, pipelines/, src/)
│       └── conftest.py               # Shared fixtures
└── dataset/
    ├── sample_claims.csv             # Inputs + expected outputs for development
    ├── claims.csv                    # Inputs only; the rows to predict
    ├── user_history.csv              # Historical claim counts and risk context
    ├── evidence_requirements.csv     # Minimum image evidence requirements
    ├── output.csv                    # Generated predictions (git-ignored)
    └── images/
        ├── sample/                   # Images referenced by sample_claims.csv
        └── test/                     # Images referenced by claims.csv
```

---

## Data File Structure

All data lives in `dataset/`. Image paths inside the CSVs are relative to `dataset/`.

| File                        | Rows | Role                                                                 |
|-----------------------------|------|----------------------------------------------------------------------|
| `sample_claims.csv`         | 20   | Labeled development set: 4 input columns plus the 10 expected outputs |
| `claims.csv`                | 44   | Input only; the rows the system must predict                         |
| `user_history.csv`          | 47   | One row per user: past claims and risk flags                         |
| `evidence_requirements.csv` | 11   | Minimum image evidence rules, by object and issue family             |
| `images/sample/case_NNN/`   | 20 cases | Images for `sample_claims.csv`                                   |
| `images/test/case_NNN/`     | 44 cases | Images for `claims.csv` (`img_1.jpg`, `img_2.jpg`, ...)          |

**`claims.csv` columns:** `user_id`, `image_paths`, `user_claim`, `claim_object`.

- `image_paths` holds one or more paths separated by `;`, e.g.
  `images/test/case_001/img_1.jpg;images/test/case_001/img_2.jpg`
- `user_claim` is a chat transcript with turns separated by ` | `. It can be multilingual or code-switched.
- `claim_object` is `car`, `laptop`, or `package`.

**`sample_claims.csv`** has the same four columns followed by the ten output columns listed under
[Input and Output Schema](#input-and-output-schema), used as the answer key.

**`user_history.csv` columns:** `user_id`, `past_claim_count`, `accept_claim`, `manual_review_claim`,
`rejected_claim`, `last_90_days_claim_count`, `history_flags` (`none`, `user_history_risk`,
`manual_review_required`, or a `;`-joined combination), `history_summary`.

**`evidence_requirements.csv` columns:** `requirement_id`, `claim_object` (`car`, `laptop`, `package`, or `all`),
`applies_to`, `minimum_image_evidence`. For each claim, the rules for `all` and for that claim's object are put in the
prompt.

---

## Installation

### Prerequisites

- Python 3.10 or newer (developed on 3.14)
- An API key for a vision-capable model served through an OpenAI-compatible endpoint (for example Gemini Flash via its
  OpenAI-compatible URL, OpenAI, or a local server)

### 1. Clone the repository

```bash
git clone https://github.com/jasonmonroe/multi-modal-evidence-review-system.git
cd multi-modal-evidence-review-system
```

### 2. Create and activate a virtual environment

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Dependencies: `numpy`, `openai`, `opencv-python`, `pandas`, `Pillow`, `python-dotenv`, plus `pytest` and
`pytest-cov` for the tests.

### 4. Configure environment variables

Copy the template and fill in your values. **Never commit `.env`** (it is git-ignored).

```bash
cp .env.example .env
```

| Variable        | Description                                                         |
|-----------------|---------------------------------------------------------------------|
| `MODEL_API_URL` | Base URL of the OpenAI-compatible API endpoint                      |
| `MODEL_NAME`    | Model identifier to call (must support image input)                 |
| `MODEL_API_KEY` | API key for the provider; read from the environment only            |

The code loads `.env` from the repository root, so keep it there.

### 5. Verify the setup

```bash
cd code
python -c "import openai, pandas, cv2, PIL, dotenv; print('Environment OK')"
```

---

## Running the System

> Dataset, `.env`, and report paths are anchored to the source files, so commands work from any directory. The
> examples below run from `code/`.

```bash
cd code

# Generate predictions for dataset/claims.csv -> dataset/output.csv
python main.py
```

Flags (combinable):

| Flag       | Effect                                                                      |
|------------|-----------------------------------------------------------------------------|
| `--sample` | Load `sample_claims.csv` instead of `claims.csv`                            |
| `--eval`   | Run the evaluation pipeline after the claims pipeline                       |
| `--eda`    | Print a dataset summary and enable image exploration                        |

Examples:

```bash
python main.py --eda            # Inspect the data before running
python main.py --sample         # Smoke-test the pipeline on the labeled sample set
python main.py --eval           # Predict on claims.csv, then evaluate on the sample set
```

The predictions are written to `dataset/output.csv`.

---

## Expected Output

A run prints a banner, then per claim the model's response, its latency, and a progress bar:

```text
----- 🖨️️ START RUN ID: 482913 🖨️️ -----

# --- Claim analysis for row 1 of 44 begins --- #
⏱ Run Time: 0m 12s 340ms
✅️☑️☑️☑️ ...	2.3%
...
# --- ✏️ Writing output to <repo>/dataset/output.csv --- #

⏱ Run Time: 9m 41s 120ms
----- 🖨️️ END RUN ID: 482913 🖨️️ -----
```

The CSV has a header row plus **one row per input row** (44 for `claims.csv`), with these 14 columns in order:

`user_id, image_paths, user_claim, claim_object, evidence_standard_met, evidence_standard_met_reason, risk_flags,
issue_type, object_part, claim_status, claim_status_justification, supporting_image_ids, valid_image, severity`

Example row (from the labeled sample set; values are what a correct prediction looks like):

```csv
user_id,image_paths,user_claim,claim_object,evidence_standard_met,evidence_standard_met_reason,risk_flags,issue_type,object_part,claim_status,claim_status_justification,supporting_image_ids,valid_image,severity
user_001,images/sample/case_001/img_1.jpg,"Customer: Hi, I found new damage ... | Support: ...",car,true,The rear bumper is visible and the dent can be verified from the submitted image.,none,dent,rear_bumper,supported,The image clearly shows a dent on the rear bumper and the user history does not add risk.,img_1,true,medium
```

Properties you can rely on:

- Every `claim_status` is `supported`, `contradicted`, or `not_enough_information`.
- `evidence_standard_met` and `valid_image` are lowercase `true` / `false`.
- `risk_flags` is `;`-joined with no spaces, sorted, or `none`.
- Users flagged `user_history_risk` in `user_history.csv` always get `user_history_risk;manual_review_required`.
- `text_instruction_present` or `possible_manipulation` forces `contradicted`, `valid_image=false`, and manual review.
- `evidence_standard_met=false` forces `not_enough_information`.
- `severity` is reset to `unknown` when the status is `not_enough_information` with a concrete severity.
- Apostrophes are stripped from text cells, and newlines in `user_claim` become ` | `.

With `--eval`, `code/evaluation/evaluation_report.md` is also written with per-strategy accuracy, latency, token use
and estimated cost, e.g.:

```text
| Strategy                | Accuracy | Latency (avg) | Cost (est) |
| Basic Prompt            | 50.00%   | 177.14s       | $0.88      |
| Chain-of-Thought Prompt | 50.00%   | 155.84s       | $0.88      |
```

---

## Running the Evaluation

The evaluation runs the system over `dataset/sample_claims.csv`, compares each predicted `claim_status` against the
labeled answer, and reports accuracy, token usage, estimated cost, and runtime for each prompting strategy
(`Basic Prompt` and `Chain-of-Thought Prompt`).

```bash
cd code
python main.py --eval
```

Outputs:

- Console summary of accuracy, tokens, cost, and timing per strategy
- Operational analysis in [code/evaluation/evaluation_report.md](code/evaluation/evaluation_report.md)

Accuracy is measured on `claim_status`. Each strategy also gets a confusion matrix (actual vs. predicted label), so
errors can be traced to specific outcomes, and the report compares both against a majority-class baseline.

---

## Results

Last recorded evaluation on the 20 labeled sample claims (model: `gemini-3.5-flash`). The full report, including
throughput and pacing, is in [code/evaluation/evaluation_report.md](code/evaluation/evaluation_report.md).

| Strategy                | Correct | Accuracy | Total Time | Avg Latency / Claim | Cost (est) |
|-------------------------|--------:|---------:|-----------:|--------------------:|-----------:|
| Basic Prompt            | 10 / 20 |   50.00% |    177.14s |               8.86s |      $0.88 |
| Chain-of-Thought Prompt | 10 / 20 |   50.00% |    155.84s |               7.79s |      $0.88 |

**Read these numbers with the class balance in mind.** The sample set is 12 `supported`, 5 `contradicted` and 3
`not_enough_information`, so a model that always answers `supported` scores 60%. Both strategies scored below that
majority-class baseline, which means the system is not yet adding value over a constant answer on this set. With only
20 samples each claim moves accuracy by 5 points, so the comparison between strategies is also not statistically
meaningful.

The next step is error analysis. The evaluation now records a confusion matrix per strategy, which shows whether the
misses are concentrated on the `contradicted` and `not_enough_information` claims, and that points to the prompt or
rule-layer changes most likely to help.

---


## Running the Unit Tests

Run from `code/` (that is where `pytest.ini` lives). No test calls the network, the real dataset or the real log
file; the OpenAI client is always mocked.

```bash
cd code
python -m pytest                                       # all tests
python -m pytest tests/src/test_utils.py               # one file
python -m pytest -k make_decision                      # tests matching a name
python -m pytest --cov=. --cov-report=term-missing     # coverage in the terminal
python -m pytest --cov=. --cov-report=html             # color-coded report
open htmlcov/index.html                                # (macOS) green = covered, red = missed
```

The tests check data types, expected keys, expected results, and inputs that must be rejected or handled safely.

---

## Input and Output Schema

**Input** (`claims.csv`): `user_id`, `image_paths` (semicolon-separated), `user_claim` (chat transcript),
`claim_object` (`car` | `laptop` | `package`).

**Output** (`output.csv`), one row per input row, columns in this order:

| Column                         | Meaning                                                                         |
|--------------------------------|---------------------------------------------------------------------------------|
| `user_id`                      | User submitting the claim                                                       |
| `image_paths`                  | Submitted image paths                                                           |
| `user_claim`                   | Claim conversation                                                              |
| `claim_object`                 | `car`, `laptop`, or `package`                                                   |
| `evidence_standard_met`        | `true` if the image set is sufficient to evaluate the claim                     |
| `evidence_standard_met_reason` | Short reason for the evidence decision                                          |
| `risk_flags`                   | Semicolon-separated flags, or `none`                                            |
| `issue_type`                   | Visible issue (`dent`, `scratch`, `crack`, `glass_shatter`, … `none`, `unknown`) |
| `object_part`                  | Relevant part (object-specific list; see problem statement)                     |
| `claim_status`                 | `supported`, `contradicted`, or `not_enough_information`                        |
| `claim_status_justification`   | Concise image-grounded explanation                                              |
| `supporting_image_ids`         | Semicolon-separated image IDs (filename without extension), or `none`           |
| `valid_image`                  | `true` if the image set is usable for automated review                          |
| `severity`                     | `none`, `low`, `medium`, `high`, or `unknown`                                   |

See [problem_statement.md](problem_statement.md) for the complete list of allowed values.

---

## Cost, Latency, and Rate Limits

`python main.py --eval` writes [code/evaluation/evaluation_report.md](code/evaluation/evaluation_report.md) with, for
each prompting strategy: accuracy, runtime, token usage, and an estimated cost. Real per-call token counts come from
each API response's `usage` field and are accumulated by `UsageTracker`, keyed by the model the provider reports.

Rate-limit handling:

- Requests are paced with a pause between claims (`PAUSE_TIMER`).
- A `RateLimitError` triggers a back-off pause (`SLEEP_TIMER`) and a retry; each retry adds `SLEEP_TIMER_INC`
  seconds of extra pacing for the rest of the run. After `MAX_RETRIES` failed attempts on one claim (for example,
  an exhausted quota) the run stops with the error instead of retrying forever.
- The OpenAI client itself retries transient failures up to 3 times with a 120-second timeout.

These values live in [code/src/constants.py](code/src/constants.py). Cost figures depend on the model and the pricing
assumed in the report, so treat them as estimates.

---

## Troubleshooting

| Symptom                                   | Likely cause / fix                                                              |
|-------------------------------------------|---------------------------------------------------------------------------------|
| `FileNotFoundError` for datasets or images | Confirm the `dataset/` folder exists at the repository root                     |
| Authentication or `None` model errors     | `.env` missing or empty; confirm `MODEL_API_URL`, `MODEL_NAME`, `MODEL_API_KEY` |
| `RateLimitError` / long pauses            | Expected on low-quota keys; the pipeline backs off and retries automatically    |
| `ModuleNotFoundError`                     | Virtual environment not activated or `pip install -r requirements.txt` not run  |

---

## Limitations and Next Steps

- **Accuracy is below the majority-class baseline** on the sample set (see [Results](#results)); improving it
  starts with the per-class error analysis.
- **Small evaluation set.** 20 labeled claims is enough to catch regressions, not to rank strategies.
- **Only `claim_status` is scored.** Per-field accuracy (issue type, object part, severity, risk flags) is not
  computed.
- **Cost is an estimate** based on a flat per-token rate, not the provider's actual pricing.
- **No response caching.** Re-running re-sends every claim to the model.

---

## Security

- Secrets are read from environment variables only and are never committed.
- `.env` and generated outputs (`dataset/output.csv`) are git-ignored.

## Author

Jason Monroe — [jason@jasonmonroe.com](mailto:jason@jasonmonroe.com)


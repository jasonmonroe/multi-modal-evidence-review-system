# Multi-Modal Evidence Review System

> **Status: work in progress.** This README is a scaffold and will be updated once the code is complete. Sections
> marked _TBD_ are placeholders.

A submission for the [HackerRank Orchestrate (June 2026) – Multi-Modal Evidence Review](https://www.hackerrank.com/contests/hackerrank-orchestrate-june26/challenges/multi-modal-review)
challenge.

The system verifies damage claims for three object types — `car`, `laptop`, and `package` — by combining:

- one or more submitted **images** (the primary source of truth)
- a short **claim conversation** (defines what needs to be checked)
- the user's **claim history** (risk context only; never overrides clear visual evidence)
- the **minimum evidence requirements** for the object and issue type

For each claim it decides whether the images **support** the claim, **contradict** it, or provide **not enough
information**, and emits a structured row of predictions (issue type, object part, severity, risk flags, supporting
image IDs, and short image-grounded justifications).

The full task description and I/O schema live in [problem_statement.md](problem_statement.md).

---

## Table of Contents

1. [How It Works](#how-it-works)
2. [Project Structure](#project-structure)
3. [Setting Up the Environment](#setting-up-the-environment)
4. [Running the System](#running-the-system)
5. [Running the Evaluation](#running-the-evaluation)
6. [Input and Output Schema](#input-and-output-schema)
7. [Cost, Latency, and Rate Limits](#cost-latency-and-rate-limits)
8. [Troubleshooting](#troubleshooting)
9. [Roadmap](#roadmap)

---

## How It Works

_TBD — final design to be documented once the code is complete._ Current flow:

1. `DataHandler` loads `claims.csv` (or `sample_claims.csv` with `--sample`), `user_history.csv`, and
   `evidence_requirements.csv`.
2. `ImageHandler` loads and prepares the images referenced by each claim row.
3. `ClaimAnalyzer` builds a prompt (claim conversation + images + user history + matching evidence requirements) and
   calls a vision-capable LLM through an OpenAI-compatible API, requesting a JSON response that conforms to
   `RESPONSE_SCHEMA`.
4. `ClaimHandler` normalizes the response to the allowed values and exports one CSV row per claim.
5. The pipeline retries on rate-limit errors with a back-off pause and paces requests between calls.
6. The evaluation pipeline runs the same analyzer over `sample_claims.csv`, compares predictions with the expected
   labels, and tracks accuracy, token usage, cost, and timing for each prompting strategy (`Basic Prompt` and
   `Chain-of-Thought Prompt`).

---

## Project Structure

```text
.
├── AGENTS.md                         # Agent/contributor rules and project contract
├── CLAUDE.md                         # Points Claude Code at AGENTS.md
├── README.md                         # You are here
├── problem_statement.md              # Full task description and I/O schema
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
│   └── tests/                        # Unit tests (mirrors agents/, models/, pipelines/, src/)
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

> The layout may shift while the project is being refactored; this tree will be refreshed with the final code.

---

## Setting Up the Environment

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

Dependencies: `numpy`, `openai`, `opencv-python`, `pandas`, `Pillow`, `python-dotenv`.

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

The code loads `../.env` relative to `code/`, so keep `.env` at the repository root.

### 5. Verify the setup

```bash
cd code
python -c "import openai, pandas, cv2, PIL, dotenv; print('Environment OK')"
```

---

## Running the System

> All commands must be run from the `code/` directory — dataset and `.env` paths are relative to it.

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

The final predictions are written to `dataset/output.csv`. Copy or move that file to the repository root as
`output.csv` for submission.

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

Run the unit tests (once populated):

```bash
cd code
python -m pytest tests/
```

> `pytest` is not in `requirements.txt` yet; install it with `pip install pytest` if you want to run the tests.

_TBD — final evaluation metrics (per-field accuracy, confusion matrix) to be added when complete._

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

_TBD — figures to be filled in from the final run._ Planned coverage in
[code/evaluation/evaluation_report.md](code/evaluation/evaluation_report.md):

- Approximate model calls for sample and test processing
- Input/output token usage and number of images processed
- Estimated cost for the full test set, with pricing assumptions
- Runtime and latency
- TPM/RPM handling: pacing between requests (`PAUSE_TIMER`), back-off on rate-limit errors (`SLEEP_TIMER`,
  `SLEEP_TIMER_INC`), and caching/batching strategy

Pacing and retry values are defined in [code/src/constants.py](code/src/constants.py).

---

## Troubleshooting

| Symptom                                   | Likely cause / fix                                                              |
|-------------------------------------------|---------------------------------------------------------------------------------|
| `FileNotFoundError` for datasets or images | Run commands from `code/`, not the repository root                              |
| Authentication or `None` model errors     | `.env` missing or empty; confirm `MODEL_API_URL`, `MODEL_NAME`, `MODEL_API_KEY` |
| `RateLimitError` / long pauses            | Expected on low-quota keys; the pipeline backs off and retries automatically    |
| `ModuleNotFoundError`                     | Virtual environment not activated or `pip install -r requirements.txt` not run  |

---

## Roadmap

- [ ] Finalize prompts and structured output schema
- [ ] Complete evaluation metrics and `evaluation_report.md`
- [ ] Add unit tests under `code/tests/`
- [ ] Add response caching to avoid repeated model calls
- [ ] Refresh this README with final architecture, results, and cost analysis

---

## Security

- Secrets are read from environment variables only and are never committed.
- `.env` and generated outputs (`dataset/output.csv`) are git-ignored.

## Author

Jason Monroe — [jason@jasonmonroe.com](mailto:jason@jasonmonroe.com)

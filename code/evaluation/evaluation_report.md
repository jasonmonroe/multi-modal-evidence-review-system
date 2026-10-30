# Operational Evaluation Report

Evaluated on `sample_claims.csv` (20 labeled claims).
Accuracy is the share of rows where the predicted `claim_status` matches the label.

## Strategy Comparison
| Strategy | Correct | Accuracy | Total Time | Avg Latency / Claim | Cost (est) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Basic Prompt | 10 / 20 | 50.00% | 177.14s | 8.86s | $0.88 |
| Chain-of-Thought Prompt | 10 / 20 | 50.00% | 155.84s | 7.79s | $0.88 |

## Baseline
A majority-class baseline that always answers `supported` scores 60.00% (12 of 20 claims). The best strategy scores 50.00%, which is 10.00 points below it.

## Operational Analysis
- **Model Used:** gemini-3.5-flash
- **Model Calls:** 40
- **Total Token Usage:** 176,824
- **Approximate Cost:** $1.77 (assumes a flat $0.01 per 1,000 tokens)
- **Throughput:** about 7.2 requests/min and 31,862 tokens/min
- **Pacing and Retries:** 1s pause between claims; on a rate-limit error the pipeline waits 20s and retries, adding 4s of extra pacing per retry

## Final Strategy Choice
We chose **Chain-of-Thought Prompt** because accuracy (50.00%) and cost were tied, and it had the shortest runtime (155.84s).

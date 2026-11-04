# summarize-and-check

Summarize a page of text, then check the summary against the source. CPU only, no GPU needed.

| Script | What it does | Needs |
|---|---|---|
| `summarize.py` | **Extractive** summary: picks the most central sentences (TextRank-style). Never invents text. | Python standard library |
| `summarize_abstractive.py` | **Abstractive** summary: a pretrained BART model writes a new summary in its own words. Long pages are chunked automatically. | `torch`, `transformers` |
| `check_summary.py` | **Rule-based checker**: flags invented numbers/names, weakly supported sentences, negation flips, opposite-word swaps, repetition, and missing key topics. | Python standard library |
| `check_summary_nli.py` | **Hybrid checker**: the rules above plus a PyTorch entailment (NLI) model that judges whether the source actually supports each summary sentence. | `torch`, `transformers` |

## Quick start

```bash
# No installs needed for these two:
python summarize.py examples/sample_page.txt -n 3
python check_summary.py examples/sample_page.txt examples/summary_bad.txt

# For the model-based scripts:
pip install -r requirements.txt
python summarize_abstractive.py examples/sample_page.txt > summary.txt
python check_summary_nli.py examples/sample_page.txt summary.txt
```

Model weights download on first use (about 1.2 GB for the summarizer, about 740 MB for the NLI checker) and are cached afterwards.

Both checkers return exit code `0` = PASS, `1` = REVIEW, `2` = FAIL, so they can gate a pipeline.

## How the checking works

- **Rules** (`check_summary.py`): every number, date, name and acronym in a summary sentence must appear in the source, otherwise the sentence is marked UNSUPPORTED. Sentences are also compared word-by-word against the best matching source passage, and negation or opposite-word differences are flagged.
- **NLI model** (`check_summary_nli.py`): for each summary sentence the closest source passages are scored as entailed / neutral / contradicted. A faithful paraphrase passes even with little word overlap; a contradicted sentence fails. The rule-based number and name checks still apply on top.

## Limits

- These checks look for **red flags**, not truth. A PASS means no red flags were found, not that the summary is verified. Read important summaries against the original.
- The entailment thresholds in `check_summary_nli.py` (0.5 to pass, below 0.2 to fail) are starting values; tune them on your own documents.
- `summarize_abstractive.py` and `check_summary_nli.py` depend on downloaded models and were not run end to end in the environment where they were written; the surrounding logic was tested with a stand-in scorer. Please report issues.
- Abstractive models can state things the source did not say. That is what the checkers are for.

## Examples

`examples/` holds a short sample page, a faithful summary of it, and a deliberately hallucinated one (invented year, dollar amount, agency and percentage, plus a flipped claim) to show what the checkers flag.

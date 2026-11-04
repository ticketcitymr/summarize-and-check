# summarize-and-check (SAR narratives)

Summarize a SAR narrative into one paragraph, check the summary against the narrative, and score the narrative against FinCEN's *Guidance on Preparing a Complete & Sufficient SAR Narrative* (Nov 2003). One file, CPU only, Python standard library.

```bash
python sar_summarizer.py narrative.txt          # summary + check + completeness
python sar_summarizer.py --batch examples/sar_good -v
python sar_summarizer.py narrative.txt --summary my_summary.txt   # check a summary you already have
```

Exit code: `0` PASS, `1` REVIEW, `2` FAIL (worst result in a batch).

## What it does
1. **Summary**: picks the sentences that carry the who / what / when / where / why / how, the totals and the follow-up action, and joins them into one paragraph (`--words` sets the cap).
2. **Check**: flags invented numbers or names, weakly supported sentences, negation flips, opposite-word swaps and repetition, and reports whether the summary kept the subject, key amount, dates, activity, follow-up action and reason for suspicion.
3. **Completeness**: scores the narrative on 8 elements from the guide (who, what, when, where, why, how, follow-up action, records location) and warns on things like "see attachment".

Optional: `--engine bart` (model rewrites the summary) and `--nli` (entailment model judges meaning). These need `pip install -r requirements.txt` and a model download, and have not been run end to end.

## Before and after
[`docs/BEFORE_AFTER.md`](docs/BEFORE_AFTER.md) shows every example narrative as the original, the one-paragraph summary, the check result and the completeness score, for reviewers and auditors. Regenerate it with `python make_report.py`.

## Examples
`examples/sar_good` (10) and `examples/sar_insufficient` (7) are the sample narratives published in the FinCEN guide (public). Do not commit real narratives to this repo.

## Limits
- The checker finds red flags, not truth. A PASS means none were found; read the summary against the original.
- Summaries are extracted sentences and can read choppy.
- The completeness score is a checklist of the guide's recommended elements, not a regulatory determination.

---
name: sar-summarize-check
description: Summarize a SAR (Suspicious Activity Report) narrative into one paragraph, verify the summary against the original for invented facts, and score the narrative against FinCEN's complete-and-sufficient guidance. Use when asked to summarize, check, or review a SAR narrative.
license: MIT
---

# SAR summarize and check

Runs `sar_summarizer.py` (Python standard library, local only) from the repo root.

## Rules
- Never paste real SAR narratives into chat, issues, or online tools. Work only on local files.
- The checker finds red flags; it cannot prove truth. Always tell the user to read the summary against the original.
- The completeness score is a checklist of FinCEN's recommended elements, not a regulatory determination.

## Steps
1. Get the path to a narrative `.txt` file. If the user pasted text, ask them to save it to a file instead.
2. Summarize and check:
   `python sar_summarizer.py <narrative.txt>`
   Use `--words 100` for a shorter summary.
3. To check a summary someone already wrote:
   `python sar_summarizer.py <narrative.txt> --summary <summary.txt>`
4. For a folder of narratives: `python sar_summarizer.py --batch <folder> -v`
5. Report back: the one-paragraph summary, the check result, and the completeness score (x/8).

## Reading the result
- Exit 0 = PASS (no red flags). Exit 1 = REVIEW (a person should look, e.g. dropped amount or subject). Exit 2 = FAIL (invented number/name or meaning flip).
- Completeness 7-8 strong, 5-6 adequate, 0-4 thin. Mention any "see attachment" warnings.
- On REVIEW or FAIL, quote the flagged sentence and explain it in plain words. Do not "fix" it silently.

## Do not
- Use `--engine bart` or `--nli` unless the user asks. They need extra installs and have not been tested end to end.
- Commit real narratives. Only the public FinCEN examples in `examples/` belong in the repo.

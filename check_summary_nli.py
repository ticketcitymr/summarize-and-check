"""
Hybrid summary checker: rules + a PyTorch entailment (NLI) model.

  * Rules (from check_summary.py) catch invented numbers, names and acronyms.
    Rules are more reliable than a neural net for these, so they always stand.
  * A pretrained NLI model judges MEANING: for each summary sentence it asks
    "does a passage in the source entail this?" and also detects contradiction.
    This is what lets a faithful paraphrase pass even when it reuses few of the
    source's words, and what catches subtle meaning changes the word checker misses.

Setup (one time):
    pip install torch transformers
    (keep this file in the same folder as check_summary.py)

Usage:
    python check_summary_nli.py source.txt summary.txt
    python summarize_abstractive.py page.txt | python check_summary_nli.py page.txt -
    python check_summary_nli.py page.txt summary.txt --model cross-encoder/nli-deberta-v3-small   # smaller

Exit code: 0 = PASS, 1 = REVIEW, 2 = FAIL.

The first run downloads the NLI model (~740 MB for the default) and caches it.
CPU is fine: a page-length summary is a few dozen short comparisons.
"""
import argparse
import sys

from check_summary import Vocab, check, content_stems, short

DEFAULT_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"


class NLI:
    """Wraps a Hugging Face NLI classifier. probs(premise, hypothesis) -> entail/neutral/contra."""

    def __init__(self, model_name):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).eval()
        self.idx = {}
        for i, label in self.model.config.id2label.items():
            label = str(label).lower()
            if "entail" in label:
                self.idx["entail"] = int(i)
            elif "contra" in label:
                self.idx["contra"] = int(i)
            elif "neutral" in label:
                self.idx["neutral"] = int(i)
        if len(self.idx) != 3:
            sys.exit(f"{model_name} does not look like an NLI model "
                     f"(labels: {list(self.model.config.id2label.values())}).")

    def probs(self, premise, hypothesis):
        enc = self.tok(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512)
        with self.torch.no_grad():
            p = self.torch.softmax(self.model(**enc).logits[0], dim=-1).tolist()
        return {k: p[i] for k, i in self.idx.items()}


def ranked_windows(sentence, src_sents, src_vocabs, k):
    """Top-k source passages (single sentences or adjacent pairs) by word overlap with the sentence."""
    stems = content_stems(sentence)
    n = len(src_sents)
    cands = []
    for i in range(n):
        for j in (i, i + 1):
            if j >= n:
                continue
            vocab = src_vocabs[i] if i == j else src_vocabs[i] | src_vocabs[j]
            score = sum(1 for s in stems if vocab.has(s)) / len(stems) if stems else 0.0
            cands.append((-score, j - i, i, j))
    cands.sort()
    return [(i, j) for _, _, i, j in cands[:k]]


def check_with_nli(source, summary, nli, top_k=4, min_entail=0.5, contra_flag=0.5, unsupported_below=0.2):
    base = check(source, summary)  # rule pass: numbers, names, coverage, repetition, length
    src_sents = base["src_sents"]
    src_vocabs = [Vocab(content_stems(s)) for s in src_sents]

    for r in base["results"]:
        best_e, best_win, max_c = 0.0, None, 0.0
        for i, j in ranked_windows(r["text"], src_sents, src_vocabs, top_k):
            p = nli.probs(" ".join(src_sents[i : j + 1]), r["text"])
            if best_win is None or p["entail"] > best_e:
                best_e, best_win = p["entail"], (i + 1, j + 1)
            max_c = max(max_c, p["contra"])

        # Keep only the hard rule findings; wording-overlap complaints no longer apply once
        # the model has judged meaning.
        issues = [x for x in r["issues"] if x.startswith(("number(s)", "name(s)"))]
        if r["hard"]:
            verdict = "UNSUPPORTED"
        elif best_e >= min_entail:
            verdict = "OK"
        elif max_c >= contra_flag and best_e < 0.3:
            verdict = "UNSUPPORTED"
            issues.append(f"contradicted by the source ({max_c:.0%} contradiction)")
        elif best_e < unsupported_below:
            verdict = "UNSUPPORTED"
            issues.append("not backed up by any source passage")
        else:
            verdict = "REVIEW"
            issues.append("only partly backed up by the source")

        r.update(verdict=verdict, issues=issues, window=best_win, entail=best_e, contra=max_c)

    results = base["results"]
    if any(r["verdict"] == "UNSUPPORTED" for r in results):
        base["overall"] = "FAIL"
    elif any(r["verdict"] == "REVIEW" for r in results) or base["notes"]:
        base["overall"] = "REVIEW"
    else:
        base["overall"] = "PASS"
    return base


def report(r):
    print(f"Source: {r['src_words']} words, {len(r['src_sents'])} sentences | "
          f"Summary: {r['sum_words']} words, {len(r['results'])} sentences "
          f"({r['sum_words'] / r['src_words']:.0%} of source)")
    print(f"Key-term coverage: {r['coverage']:.0%}"
          + (f"  (missing: {', '.join(r['missing'][:6])})" if r["missing"] else ""))
    print()
    for x in r["results"]:
        a, b = x["window"]
        where = f"source sentence {a}" if a == b else f"source sentences {a}-{b}"
        print(f"[{x['n']}] {x['verdict']:<11} entailed {x['entail']:.0%} / contradicted {x['contra']:.0%}  "
              f"(best match: {where})")
        print(f"    \"{short(x['text'])}\"")
        for issue in x["issues"]:
            print(f"    - {issue}")
        if x["verdict"] != "OK":
            print(f"    closest source: \"{short(' '.join(r['src_sents'][a - 1:b]))}\"")
        print()
    for n in r["notes"]:
        print(f"NOTE: {n}")
    bad = sum(1 for x in r["results"] if x["verdict"] == "UNSUPPORTED")
    rev = sum(1 for x in r["results"] if x["verdict"] == "REVIEW")
    print(f"OVERALL: {r['overall']}  ({bad} unsupported, {rev} to review, "
          f"{len(r['results']) - bad - rev} ok)")


def main():
    ap = argparse.ArgumentParser(description="Check a summary against its source with rules + an NLI model.")
    ap.add_argument("source", help="original text file (or - for stdin)")
    ap.add_argument("summary", help="summary text file (or - for stdin)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"NLI model id (default {DEFAULT_MODEL})")
    ap.add_argument("--top-k", type=int, default=4, help="source passages compared per summary sentence (default 4)")
    ap.add_argument("--min-entail", type=float, default=0.5, help="entailment needed for OK (default 0.5)")
    args = ap.parse_args()
    if args.source == "-" and args.summary == "-":
        sys.exit("Only one of source/summary can be read from stdin.")

    def read(p):
        return sys.stdin.read() if p == "-" else open(p, encoding="utf-8").read()

    r = check_with_nli(read(args.source), read(args.summary), NLI(args.model),
                       top_k=args.top_k, min_entail=args.min_entail)
    report(r)
    sys.exit({"PASS": 0, "REVIEW": 1, "FAIL": 2}[r["overall"]])


if __name__ == "__main__":
    main()

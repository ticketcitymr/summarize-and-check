"""
Summary checker: compares a summary against its source text and flags claims the
source does not support. Pure Python standard library, no installs, no model.

It is built to catch the usual abstractive-summary failures:
  * invented numbers, dates, dollar amounts, percentages
  * invented names, organizations, acronyms
  * sentences with little wording support anywhere in the source
  * negation flips ("must" -> "never needs to") and opposite-word swaps
  * repetition, and key topics from the source that the summary left out

Usage:
    python check_summary.py source.txt summary.txt
    python summarize_abstractive.py page.txt | python check_summary.py page.txt -

Exit code: 0 = PASS, 1 = REVIEW (look at flagged sentences), 2 = FAIL (unsupported claims).

LIMIT: this is a wording-and-facts check. It can prove a summary contains things the
source never mentioned. It cannot prove a summary is TRUE, and it can miss subtle
meaning changes that reuse the source's own words. Treat PASS as "no red flags".
"""
import argparse
import re
import sys
from collections import Counter

STOPWORDS = set("""
a about above after again against all am an and any are as at be because been before
being below between both but by can could did do does doing down during each few for
from further had has have having he her here hers him his how i if in into is it its
just me more most my no nor not now of off on once only or other our out over own same
she should so some such than that the their them then there these they this those
through to too under until up very was we were what when where which while who whom
why will with would you your also may might must shall us via per
""".split())

NEGATIONS = {"not", "no", "never", "none", "neither", "nor", "cannot", "without", "n't", "nothing"}

# Opposite pairs: if the summary uses one and the matched source sentence uses the other, flag it.
OPPOSITES = [
    ("increase", "decrease"), ("rise", "fall"), ("more", "less"), ("higher", "lower"),
    ("approve", "reject"), ("allow", "prohibit"), ("required", "optional"),
    ("effective", "ineffective"), ("before", "after"), ("legal", "illegal"),
    ("valid", "invalid"), ("strong", "weak"), ("grow", "shrink"), ("gain", "loss"),
    ("success", "failure"), ("improve", "worsen"), ("accept", "refuse"), ("true", "false"),
]


def split_sentences(text):
    text = re.sub(r"\s+", " ", text.strip())
    text = re.sub(r"\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc|e\.g|i\.e|U\.S)\.", r"\1<DOT>", text)
    parts = re.split(r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9])", text)
    return [p.replace("<DOT>", ".").strip() for p in parts if p.strip()]


def stem(w):
    w = w.lower()
    if w.endswith("ies") and len(w) > 5:
        return w[:-3] + "y"
    for suf in ("ization", "ations", "ation", "ments", "ment", "ings", "ing", "edly", "ed", "es", "ly", "s"):
        if len(w) > len(suf) + 2 and w.endswith(suf):
            w = w[: -len(suf)]
            break
    if w.endswith("e") and len(w) > 4:
        w = w[:-1]
    return w


def content_stems(text):
    toks = re.findall(r"[a-z']+", text.lower())  # letters only: numbers are handled separately
    return [stem(t) for t in toks if t not in STOPWORDS and len(t) > 1]


def stem_to_word(text):
    """Map each stem back to the word as written, so reports show real words."""
    m = {}
    for t in re.findall(r"[A-Za-z']+", text):
        if t.lower() not in STOPWORDS and len(t) > 1:
            m.setdefault(stem(t), t)
    return m


def numbers(text):
    """Numeric cores like 10000, 40, 1970 (commas, $ and % ignored for comparison)."""
    out = set()
    for m in re.findall(r"\d[\d,]*(?:\.\d+)?", text):
        out.add(m.replace(",", "").rstrip("."))
    return out


def entities(sentence):
    """Capitalized words not at the start of the sentence, plus ALL-CAPS acronyms anywhere."""
    toks = re.findall(r"[A-Za-z][A-Za-z'-]*", sentence)
    found = []
    for i, t in enumerate(toks):
        if t.lower() in STOPWORDS:
            continue
        if t.isupper() and len(t) > 1:
            found.append(t)
        elif i > 0 and t[0].isupper():
            found.append(t)
    return found


class Vocab:
    """Stem set plus 5-letter prefixes, so 'validation' still matches 'validated'."""

    def __init__(self, stems):
        self.stems = set(stems)
        self.prefixes = {s[:5] for s in self.stems if len(s) >= 6}

    def has(self, s):
        return s in self.stems or (len(s) >= 6 and s[:5] in self.prefixes) or (len(s) >= 5 and s in self.prefixes)

    def __or__(self, other):
        return Vocab(self.stems | other.stems)


def best_window(summ_stems, src_vocabs):
    """Best single source sentence or adjacent pair covering the summary sentence's words."""
    best = (0.0, (0, 0))
    n = len(src_vocabs)
    for i in range(n):
        for j in (i, i + 1):
            if j >= n:
                continue
            vocab = src_vocabs[i] if i == j else src_vocabs[i] | src_vocabs[j]
            score = sum(1 for s in summ_stems if vocab.has(s)) / len(summ_stems)
            if score > best[0] + 1e-9:
                best = (score, (i, j))
    return best


def check(source, summary, min_support=0.7):
    src_sents = split_sentences(source)
    sum_sents = split_sentences(summary)
    if not src_sents or not sum_sents:
        sys.exit("Source and summary must both contain text.")

    src_stem_lists = [content_stems(s) for s in src_sents]
    src_vocabs = [Vocab(l) for l in src_stem_lists]
    global_vocab = Vocab([s for l in src_stem_lists for s in l])
    src_numbers = numbers(source)
    src_tokens_lower = set(re.findall(r"[A-Za-z][A-Za-z'-]*", source.lower()))
    src_text_lower = source.lower()

    results = []
    for k, sent in enumerate(sum_sents, 1):
        stems = content_stems(sent)
        issues, hard = [], False

        # Invented numbers
        bad_nums = sorted(n for n in numbers(sent) if n not in src_numbers)
        if bad_nums:
            issues.append(f"number(s) not in source: {', '.join(bad_nums)}")
            hard = True

        # Invented names / acronyms
        bad_ents = sorted({e for e in entities(sent) if e.lower() not in src_tokens_lower})
        if bad_ents:
            issues.append(f"name(s) not in source: {', '.join(bad_ents)}")
            hard = True

        if stems:
            support, (i, j) = best_window(stems, src_vocabs)
            novel = sorted({s for s in stems if not global_vocab.has(s)})
        else:
            support, (i, j), novel = 1.0, (0, 0), []

        window_text = " ".join(src_sents[i : j + 1]).lower()
        window_vocab = Vocab(content_stems(window_text))

        # Negation flip
        neg_sum = bool(NEGATIONS & set(re.findall(r"[a-z']+", sent.lower())))
        neg_src = bool(NEGATIONS & set(re.findall(r"[a-z']+", window_text)))
        negation_flag = neg_sum != neg_src
        if negation_flag:
            issues.append("negation differs from the closest source sentence")

        # Opposite-word swap
        sum_stems_set = set(stems)
        for a, b in OPPOSITES:
            for x, y in ((a, b), (b, a)):
                if stem(x) in sum_stems_set and not window_vocab.has(stem(x)) and window_vocab.has(stem(y)):
                    issues.append(f"possible reversal: summary says '{x}', source says '{y}'")

        if novel:
            shown = stem_to_word(sent)
            issues.append(f"wording not found in source: {', '.join(shown.get(s, s) for s in novel[:6])}")

        if support < 0.5:
            verdict = "UNSUPPORTED"
            issues.insert(0, f"only {support:.0%} of its key words appear near any source passage")
        elif hard:
            verdict = "UNSUPPORTED"
        elif negation_flag or support < min_support or any("reversal" in x for x in issues) or len(novel) >= 2:
            verdict = "REVIEW"
        else:
            verdict = "OK"

        results.append({
            "n": k, "text": sent, "verdict": verdict, "support": support, "hard": hard,
            "window": (i + 1, j + 1), "issues": issues,
        })

    # Whole-summary checks
    src_words, sum_words = len(source.split()), len(summary.split())
    key = [w for w, _ in Counter(s for l in src_stem_lists for s in l).most_common(12)]
    sum_vocab = Vocab(content_stems(summary))
    covered = [w for w in key if sum_vocab.has(w)]
    missing = [w for w in key if not sum_vocab.has(w)]
    coverage = len(covered) / len(key) if key else 1.0

    words = re.findall(r"[a-z']+", summary.lower())
    trigrams = [tuple(words[i : i + 3]) for i in range(len(words) - 2)]
    repeated = sum(c - 1 for c in Counter(trigrams).values() if c > 1)
    rep_ratio = repeated / len(trigrams) if trigrams else 0.0

    notes = []
    if rep_ratio > 0.15:
        notes.append(f"repetition: {rep_ratio:.0%} of word triples repeat")
    if coverage < 0.3:
        notes.append(f"covers only {coverage:.0%} of the source's top terms")
    if sum_words > 0.6 * src_words:
        notes.append(f"summary is {sum_words / src_words:.0%} of the source length (barely shorter)")

    if any(r["verdict"] == "UNSUPPORTED" for r in results):
        overall = "FAIL"
    elif any(r["verdict"] == "REVIEW" for r in results) or notes:
        overall = "REVIEW"
    else:
        overall = "PASS"

    return {
        "results": results, "overall": overall, "notes": notes, "coverage": coverage,
        "covered": covered, "missing": missing, "src_words": src_words, "sum_words": sum_words,
        "src_sents": src_sents,
    }


def short(s, n=110):
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


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
        print(f"[{x['n']}] {x['verdict']:<11} support {x['support']:.0%}  (best match: {where})")
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
    ap = argparse.ArgumentParser(description="Check a summary against its source text.")
    ap.add_argument("source", help="original text file (or - for stdin)")
    ap.add_argument("summary", help="summary text file (or - for stdin)")
    ap.add_argument("--min-support", type=float, default=0.7,
                    help="sentences below this wording support are marked REVIEW (default 0.7)")
    args = ap.parse_args()
    if args.source == "-" and args.summary == "-":
        sys.exit("Only one of source/summary can be read from stdin.")

    def read(p):
        return sys.stdin.read() if p == "-" else open(p, encoding="utf-8").read()

    r = check(read(args.source), read(args.summary), args.min_support)
    report(r)
    sys.exit({"PASS": 0, "REVIEW": 1, "FAIL": 2}[r["overall"]])


if __name__ == "__main__":
    main()

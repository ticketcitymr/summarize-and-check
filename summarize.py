"""
Extractive text summarizer. Pure Python standard library, no installs, CPU only.

Method (TextRank-style):
  1. Split the text into sentences.
  2. Turn each sentence into a bag of meaningful words (stopwords removed).
  3. Score every sentence by how similar it is to all the other sentences
     (a sentence that shares key terms with many others is central to the text).
  4. Add a small bonus for sentences near the start.
  5. Keep the top N and print them in their original order.

Usage:
  python summarize.py page.txt              # summarize a file (3 sentences)
  python summarize.py page.txt -n 5         # 5 sentences
  python summarize.py page.txt -r 0.25      # keep about 25% of the sentences
  cat page.txt | python summarize.py -      # read from stdin
"""
import argparse
import math
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


def split_sentences(text):
    text = re.sub(r"\s+", " ", text.strip())
    # Split after . ! ? followed by space + capital/quote/digit; protect common abbreviations.
    text = re.sub(r"\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc|e\.g|i\.e|U\.S)\.", r"\1<DOT>", text)
    parts = re.split(r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9])", text)
    return [p.replace("<DOT>", ".").strip() for p in parts if p.strip()]


def words(sentence):
    return [w for w in re.findall(r"[a-z0-9']+", sentence.lower())
            if w not in STOPWORDS and len(w) > 1]


def cosine(a, b):
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    dot = sum(a[w] * b[w] for w in common)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def rank_sentences(sentences, damping=0.85, iterations=50):
    n = len(sentences)
    # Weight words by inverse document frequency across sentences, so rare
    # informative words count more than words sprinkled everywhere.
    bags = [Counter(words(s)) for s in sentences]
    df = Counter(w for bag in bags for w in bag)
    vecs = [{w: c * math.log(1 + n / df[w]) for w, c in bag.items()} for bag in bags]

    sim = [[cosine(vecs[i], vecs[j]) if i != j else 0.0 for j in range(n)] for i in range(n)]
    out_sum = [sum(row) or 1.0 for row in sim]

    score = [1.0 / n] * n
    for _ in range(iterations):
        score = [
            (1 - damping) / n
            + damping * sum(sim[j][i] / out_sum[j] * score[j] for j in range(n))
            for i in range(n)
        ]
    return score


def summarize(text, num_sentences=3, ratio=None):
    sentences = split_sentences(text)
    if len(sentences) <= num_sentences and ratio is None:
        return " ".join(sentences)

    if ratio is not None:
        num_sentences = max(1, round(len(sentences) * ratio))
    num_sentences = min(num_sentences, len(sentences))

    score = rank_sentences(sentences)

    # Lead bias: opening sentences of a page usually carry the topic.
    for i in range(len(score)):
        score[i] *= 1.0 + 0.25 / (1 + i)
    # Very short fragments rarely make good summary sentences.
    for i, s in enumerate(sentences):
        if len(s.split()) < 5:
            score[i] *= 0.5

    top = sorted(sorted(range(len(sentences)), key=lambda i: score[i], reverse=True)[:num_sentences])
    return " ".join(sentences[i] for i in top)


def main():
    ap = argparse.ArgumentParser(description="Summarize a page of text.")
    ap.add_argument("file", help="text file to summarize, or - for stdin")
    ap.add_argument("-n", "--sentences", type=int, default=3, help="sentences to keep (default 3)")
    ap.add_argument("-r", "--ratio", type=float, help="fraction of sentences to keep (overrides -n)")
    args = ap.parse_args()

    text = sys.stdin.read() if args.file == "-" else open(args.file, encoding="utf-8").read()
    if not text.strip():
        sys.exit("No text to summarize.")
    print(summarize(text, args.sentences, args.ratio))


if __name__ == "__main__":
    main()

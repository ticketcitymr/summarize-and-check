"""
Abstractive text summarizer: a pretrained seq2seq model WRITES a new summary
in its own words (unlike summarize.py, which only picks existing sentences).

CPU only, no GPU needed. First run downloads the model once (~1.2 GB for the
default) and caches it; after that it works offline.

Setup (one time):
    pip install torch transformers

Usage:
    python summarize_abstractive.py page.txt
    python summarize_abstractive.py page.txt --max-tokens 80      # shorter
    python summarize_abstractive.py page.txt --max-tokens 200     # longer
    cat page.txt | python summarize_abstractive.py -
    python summarize_abstractive.py page.txt --model sshleifer/distilbart-cnn-6-6   # smaller/faster
    python summarize_abstractive.py page.txt --model t5-small                       # tiny, lower quality

Long pages are split into sentence-aligned chunks that fit the model's input
limit, each chunk is summarized, and the partial summaries are summarized once
more into a final result.
"""
import argparse
import re
import sys

DEFAULT_MODEL = "sshleifer/distilbart-cnn-12-6"  # news-trained BART, good general quality on CPU


def split_sentences(text):
    text = re.sub(r"\s+", " ", text.strip())
    text = re.sub(r"\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc|e\.g|i\.e|U\.S)\.", r"\1<DOT>", text)
    parts = re.split(r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9])", text)
    return [p.replace("<DOT>", ".").strip() for p in parts if p.strip()]


def chunk_sentences(sentences, tokenizer, budget):
    """Group sentences into chunks of at most `budget` tokens, never splitting a sentence."""
    chunks, cur, cur_len = [], [], 0
    for s in sentences:
        n = len(tokenizer.encode(s, add_special_tokens=False))
        if cur and cur_len + n > budget:
            chunks.append(" ".join(cur))
            cur, cur_len = [], 0
        cur.append(s)
        cur_len += n
    if cur:
        chunks.append(" ".join(cur))
    return chunks


class Summarizer:
    def __init__(self, model_name, beams=4):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.torch = torch
        self.beams = beams
        self.is_t5 = "t5" in model_name.lower()
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).eval()
        limit = self.tok.model_max_length
        self.input_limit = min(limit if limit and limit < 100_000 else 1024, 1024)

    def _generate(self, text, max_tokens, min_tokens):
        if self.is_t5:
            text = "summarize: " + text
        enc = self.tok(text, return_tensors="pt", truncation=True, max_length=self.input_limit)
        in_len = enc["input_ids"].shape[1]
        # Never ask for a summary longer than the input; keep min below max.
        max_len = max(10, min(max_tokens, in_len))
        min_len = max(5, min(min_tokens, max_len // 2, in_len // 2))
        with self.torch.no_grad():
            out = self.model.generate(
                **enc,
                max_length=max_len,
                min_length=min_len,
                num_beams=self.beams,
                no_repeat_ngram_size=3,
                length_penalty=2.0,
                early_stopping=True,
            )
        return self.tok.decode(out[0], skip_special_tokens=True).strip()

    def summarize(self, text, max_tokens=130, min_tokens=30):
        sentences = split_sentences(text)
        if not sentences:
            return ""
        # Leave headroom under the model's input limit for special tokens.
        chunks = chunk_sentences(sentences, self.tok, self.input_limit - 50)
        partials = [self._generate(c, max_tokens, min_tokens) for c in chunks]
        if len(partials) == 1:
            return partials[0]
        combined = " ".join(partials)
        return self._generate(combined, max_tokens, min_tokens)


def main():
    ap = argparse.ArgumentParser(description="Abstractive summary of a page of text.")
    ap.add_argument("file", help="text file to summarize, or - for stdin")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"Hugging Face model id (default {DEFAULT_MODEL})")
    ap.add_argument("--max-tokens", type=int, default=130, help="upper bound on summary length in tokens (default 130, roughly 100 words)")
    ap.add_argument("--min-tokens", type=int, default=30, help="lower bound on summary length in tokens (default 30)")
    ap.add_argument("--beams", type=int, default=4, help="beam search width; 1 is fastest, 4 is higher quality")
    args = ap.parse_args()

    text = sys.stdin.read() if args.file == "-" else open(args.file, encoding="utf-8").read()
    if not text.strip():
        sys.exit("No text to summarize.")

    print(Summarizer(args.model, args.beams).summarize(text, args.max_tokens, args.min_tokens))


if __name__ == "__main__":
    main()

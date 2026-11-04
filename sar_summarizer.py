#!/usr/bin/env python3
"""
sar_summarizer.py - summarize a SAR narrative into ONE paragraph, then check the summary.

Everything runs locally. Nothing is sent anywhere. (Real SAR narratives are confidential:
keep it that way. Do not paste them into online tools.)

PIPELINE
  1. SUMMARIZE  the narrative into one paragraph.
       default engine  "guided": picks the real sentences that carry the FinCEN "5 W's + how"
                       (who / what / when / where / why / how) plus the amounts and the bank's
                       follow-up action. Extractive, so it cannot invent a number or a name.
       optional engine "bart":   a small pretrained model rewrites it in new words. Needs
                       `pip install torch transformers`. Abstractive models CAN invent facts,
                       which is exactly what step 2 is for.
  2. CHECK      every summary sentence against the narrative:
                invented numbers/dates/amounts, invented names/acronyms, weak wording support,
                negation flips, opposite-word swaps, repetition, and whether the summary kept
                the key facts (subject, headline amount, period, activity, follow-up action).
                Optional --nli adds a PyTorch entailment model for meaning-level checks.
  3. COMPLETENESS of the narrative itself, against FinCEN's "Guidance on Preparing a Complete
                & Sufficient SAR Narrative" (Nov 2003): who, what, when, where, why, how,
                follow-up action, and where records are kept; flags "see attachment" etc.

USAGE
  python sar_summarizer.py narrative.txt                 # summarize + check + completeness
  python sar_summarizer.py narrative.txt --words 100     # shorter summary
  python sar_summarizer.py narrative.txt --summary mine.txt   # check a summary someone wrote
  python sar_summarizer.py --batch examples/sar_good/    # run every .txt file in a folder
  cat narrative.txt | python sar_summarizer.py -
  python sar_summarizer.py narrative.txt --engine bart [--nli]

EXIT CODE  0 = PASS, 1 = REVIEW, 2 = FAIL (worst result, in batch mode)

LIMITS  The checker finds red flags; it cannot prove a summary is true. A PASS means no red
        flags. A human must still read the summary against the narrative before relying on it.
"""
import argparse
import math
import re
import sys
import textwrap
from collections import Counter
from pathlib import Path

# =============================================================================
# 1. Text utilities
# =============================================================================
STOPWORDS = set("""
a about above after again against all am an and any are as at be because been before
being below between both but by can could did do does doing down during each few for
from further had has have having he her here hers him his how i if in into is it its
just me more most my no nor not now of off on once only or other our out over own same
she should so some such than that the their them then there these they this those
through to too under until up very was we were what when where which while who whom
why will with would you your also may might must shall us via per
""".split())


def clean(text):
    """Normalize typographic characters so tokenizing behaves the same on every machine."""
    for a, b in (("’", "'"), ("‘", "'"), ("“", '"'), ("”", '"'),
                 (" ", " "), ("–", "-"), ("—", "-")):
        text = text.replace(a, b)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


_ABBR = r"Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|Ave|Blvd|vs|etc|e\.g|i\.e|U\.S|Inc|Corp|Co|Ltd|LLC|No|Esq|ESQ|approx|Dept|N\.A"


def split_sentences(text):
    t = re.sub(r"\s+", " ", text.strip())
    t = re.sub(r"(^|[.!?;:]\s+|\b(?:and|or)\s+)(\d{1,2})\.\s+(?=[A-Z0-9])", r"\1(\2) ", t)   # "2. Foo" list markers
    t = re.sub(rf"\b({_ABBR})\.", r"\1<DOT>", t)
    parts = re.split(r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9])", t)
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
    return [stem(t) for t in re.findall(r"[a-z]+", text.lower()) if t not in STOPWORDS and len(t) > 1]


class Vocab:
    """Stem set plus 5-letter prefixes, so 'validation' still matches 'validated'."""

    def __init__(self, stems):
        self.stems = set(stems)
        self.prefixes = {s[:5] for s in self.stems if len(s) >= 6}

    def has(self, s):
        return s in self.stems or (len(s) >= 6 and s[:5] in self.prefixes) or (len(s) >= 5 and s in self.prefixes)

    def __or__(self, other):
        return Vocab(self.stems | other.stems)


def cosine(a, b):
    common = set(a) & set(b)
    if not common:
        return 0.0
    dot = sum(a[w] * b[w] for w in common)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def rank_sentences(sents, damping=0.85, iters=50):
    """TextRank: a sentence is central if it shares key terms with many others. Returns (scores, similarity)."""
    n = len(sents)
    bags = [Counter(content_stems(s)) for s in sents]
    df = Counter(w for b in bags for w in b)
    vecs = [{w: c * math.log(1 + n / df[w]) for w, c in b.items()} for b in bags]
    sim = [[cosine(vecs[i], vecs[j]) if i != j else 0.0 for j in range(n)] for i in range(n)]
    outs = [sum(r) or 1.0 for r in sim]
    score = [1.0 / n] * n
    for _ in range(iters):
        score = [(1 - damping) / n + damping * sum(sim[j][i] / outs[j] * score[j] for j in range(n))
                 for i in range(n)]
    return score, sim


# =============================================================================
# 2. SAR knowledge: what the FinCEN guide says a narrative should contain
# =============================================================================
_MON = (r"(?:January|February|March|April|May|June|July|August|September|October|November|December"
        r"|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept|Sep|Oct|Nov|Dec)")
DATE_RE = re.compile(
    rf"\b{_MON}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?\b"      # May 1, 2003 / March 21
    rf"|\b{_MON}\.?\s+(?:of\s+)?\d{{4}}\b"                              # March of 1994 / June 2003
    rf"|\b\d{{1,2}}[/-]\d{{1,2}}(?:[/-]\d{{2,4}})?\b"                   # 03/15/02, 06/03, 12-19-02
    rf"|\b(?:19|20)\d{{2}}\b",                                          # bare year
    re.I)
AMOUNT_RE = re.compile(r"\$\s?(\d[\d,]*(?:\.\d+)?)")

WHY_RE = re.compile(
    r"\b(suspicio\w*|suspected|suspect nature|(?:to|i|we) suspect|unusual|structur\w*|circumvent\w*|evad\w+|avoid\w*|inconsistent|"
    r"commensurate|money laundering|illegal|legitimate|unable to (?:determine|substantiate)|"
    r"not a licensed|unlicensed|tax evasion|is being (?:filed|reported)|being reported|reported for|"
    r"filed (?:due|because)|this sar|this suspicious activity report)\b", re.I)
ACTIVITY_RE = re.compile(
    r"\b(deposit\w*|wires?|withdraw\w*|checks?|money orders?|cash\w*|chips?|transfers?|purchas\w*|"
    r"remitt\w*|transmittals?|transact\w*|posted)\b", re.I)
WHO_RE = re.compile(
    r"\b(customers?|owner|signer|student|suspects?|subjects?|patrons?|clients?|sender|originator|"
    r"beneficiar\w*|employed|account holder|power of attorney|companion)\b", re.I)
WHERE_KW = re.compile(r"\b(branch\w*|office|store|casino|cage|located|location|address|street|foreign|"
                      r"overseas|abroad|country|countries|city)\b", re.I)
_PLACES = ("Mexico|Canada|China|Hong Kong|Dubai|UAE|United Arab Emirates|Iran|Iraq|Yemen|Pakistan|India|"
           "Afghanistan|Russia|Ukraine|Colombia|Venezuela|Panama|Cayman|Bahamas|Antigua|Jamaica|Haiti|Nigeria|"
           "Ghana|Egypt|Lebanon|Turkey|Syria|Israel|Saudi Arabia|Cuba|Brazil|Argentina|Peru|Ecuador|Guatemala|"
           "El Salvador|Honduras|Nicaragua|Costa Rica|Dominican Republic|Philippines|Vietnam|Thailand|Korea|"
           "Japan|Taiwan|Singapore|Malaysia|Indonesia|Australia|New Zealand|England|United Kingdom|France|"
           "Germany|Italy|Spain|Poland|Switzerland|Middle East|Europe|Africa|Asia|"
           "Alabama|Alaska|Arizona|Arkansas|California|Colorado|Connecticut|Delaware|Florida|Georgia|Hawaii|"
           "Idaho|Illinois|Indiana|Iowa|Kansas|Kentucky|Louisiana|Maine|Maryland|Massachusetts|Michigan|"
           "Minnesota|Mississippi|Missouri|Montana|Nebraska|Nevada|New Hampshire|New Jersey|New Mexico|"
           "New York|North Carolina|North Dakota|Ohio|Oklahoma|Oregon|Pennsylvania|Rhode Island|"
           "South Carolina|South Dakota|Tennessee|Texas|Utah|Vermont|Virginia|Washington|West Virginia|"
           "Wisconsin|Wyoming")
WHERE_PLACE = re.compile(rf"\b(?:{_PLACES})\b|,\s[A-Z]{{2}}\b")
# Follow-up action BY THE FILER: the filing institution (we / our bank / management) is the subject,
# or the sentence states the disposition outright. A third party closing its own account does not count.
_FILER = (r"(?:we|our (?:\w+ )?(?:bank|firm|casino|institution)|the (?:\w+ )?(?:bank|firm|casino|institution)"
          r"|(?:bank )?management|BM)")
_ACT_VERBS = (r"clos\w+|terminat\w+|restrict\w+|monitor\w*|bar(?:red)?|refer\w*|end\w*|exit\w*|elect\w*|"
              r"decid\w*|continue\w*|refus\w*")
ACTION_RE = re.compile(
    r"\b" + _FILER + r"\s+(?:(?:has|have|had|will|is|are|also)\s+){0,2}(?:\w+\s+){0,2}?(?:" + _ACT_VERBS + r")\b"
    r"|\baccounts? (?:has |have |had |was |were |is |are |will )?(?:been |being |be )*(?:clos|restrict|terminat)\w*"
    r"|\b(?:has|have) (?:elected|decided) to\b"
    r"|\binvestigat\w*(?: \w+){0,2} (?:is|are) (?:continuing|ongoing)\b"
    r"|\bsupplemental sar\b|\bbarred from\b|\breferred to\b|\bcontact(?:ed)? law enforcement\b"
    r"|\bdecided to end\b|\bend our banking\b|\b(?:was|were) (?:terminated|dismissed|fired)\b", re.I)
RECORDS_RE = re.compile(
    r"\b(records?|documentation|case (?:file|number|#)|file number|maintained|retained|reference number|"
    r"investigation case)\b", re.I)
HOW_RE = re.compile(
    r"\b(followed by|immediately|then|after|before|subsequent\w*|through|via|using|by (?:wire|check|cash|means)|"
    r"remote|online|internet|night deposits?|same day|next day|day after|consecutive|pattern|repeat\w*|"
    r"each (?:week|day|time|deposit)|always|usually|bi-weekly|aggregate)\b", re.I)
ATTACH_RE = re.compile(r"\bsee (?:the )?attach\w*|\battached\b|\battachment\b|\benclosed\b", re.I)

TYPOLOGIES = [
    ("structuring", r"structur\w+|circumvent\w*|evad\w+ (?:the )?(?:reporting|ctr)|avoid\w* (?:a |the |filing )?(?:ctr|currency)|(?:just |slightly )?(?:under|below) \$10,000"),
    ("wire transfers", r"\bwires?\b|wire transfers?"),
    ("cash deposits", r"cash deposits?|bulk cash|night deposits?|cash-in|cashed"),
    ("money orders", r"money orders?"),
    ("third-party checks", r"third[- ]party"),
    ("check activity", r"\bchecks?\b"),
    ("informal value transfer", r"informal value transfer|hawala|ivts"),
    ("layering / correspondent accounts", r"layering|pass[- ]through|correspondent"),
    ("no legitimate business", r"no (?:indication of )?legitimate|not appear to maintain|unable to substantiate|not (?:a )?licensed|unlicensed"),
    ("casino chips / gaming", r"\bchips?\b|black ?jack|pit (?:boss|supervisor|clerks?)|\bcage\b"),
    ("foreign jurisdiction", r"\bforeign\b|overseas|\babroad\b"),
    ("tax evasion", r"tax evasion|pay taxes|avoid taxes"),
    ("securities / brokerage", r"brokerage|securities"),
]
TYPOLOGY_RES = [(label, re.compile(rx, re.I)) for label, rx in TYPOLOGIES]

# Capitalized-word runs = candidate names. Filter the usual sentence-opening words.
_CAP = r"[A-Z][A-Za-z'&-]*"
NAME_RE = re.compile(rf"{_CAP}(?:\s+(?:of|de|the|and|&)\s+{_CAP}|\s+{_CAP})*(?:,?\s+(?:Inc|Corp|Ltd|LLC|Co)\.?)?")
NAME_STOP = set("""The This That These Those Our We Its It Their They He She His Her On In At By For From To Between
During After Before Also Additionally However Therefore Specifically Because Due As If When Where While Since Until
Prior Later Today All Many Some A An One Two Both Each Any No Not But And Or Per Reviews Review Investigation
Investigative Account Accounts Mr Ms Mrs Dr Immediately Periodic Individual Attempted Contact Specifically Once
Doe's Customer One Two Three Four Five Six Seven Eight Nine Ten Checks Wire Cash Funds SAR CTR CTRs IRS OFAC""".split())
_MONTHS_DAYS = set("""january february march april may june july august september october november december
monday tuesday wednesday thursday friday saturday sunday""".split())


def find_names(text):
    """Counter of probable proper names (people, businesses, banks) found in the text."""
    found = Counter()
    for sent in split_sentences(text):
        for m in NAME_RE.finditer(sent):
            toks = m.group().replace(",", "").split()
            while toks and (toks[0] in NAME_STOP or toks[0].lower() in _MONTHS_DAYS):
                toks.pop(0)
            if not toks:
                continue
            if m.start() == 0 and len(toks) == len(m.group().replace(",", "").split()) and len(toks) == 1:
                continue                                  # lone capitalized word opening a sentence
            if len(toks) == 1 and (toks[0] in NAME_STOP or toks[0].isupper() or toks[0].lower() in _MONTHS_DAYS):
                continue
            found[" ".join(toks)] += 1
    return found


def top_names(text, k=3):
    """The most frequently mentioned names in the narrative (the likely subjects)."""
    names = find_names(text)
    return [n for n, _ in sorted(names.items(), key=lambda kv: (-kv[1], -len(kv[0])))[:k]]


def amounts(text):
    out = []
    for m in AMOUNT_RE.finditer(text):
        try:
            out.append(float(m.group(1).replace(",", "")))
        except ValueError:
            pass
    return out


TOTAL_RE = re.compile(r"\b(?:totall?ing|total(?:s|ed)?|aggregate[sd]?)\b[^$]{0,25}\$\s?(\d[\d,]*(?:\.\d+)?)", re.I)
REPORTING_THRESHOLDS = {10000.0, 3000.0}      # CTR / recordkeeping thresholds: rules, not activity amounts


def key_amounts(text):
    """The figures a reader needs: amounts introduced as totals; else the largest non-threshold amounts."""
    totals = {float(m.group(1).replace(",", "")) for m in TOTAL_RE.finditer(text)}
    if totals:
        return sorted(totals, reverse=True)
    rest = sorted({v for v in amounts(text) if v not in REPORTING_THRESHOLDS}, reverse=True)
    return rest[:3]


def fmt_money(v):
    return f"${v:,.0f}" if abs(v - round(v)) < 1e-9 else f"${v:,.2f}"


def typologies(text):
    return [label for label, rx in TYPOLOGY_RES if rx.search(text)]


def sentence_roles(sent):
    """Which FinCEN elements does this sentence carry?"""
    roles = set()
    why_cues = {m.group(0).lower() for m in WHY_RE.finditer(sent)}
    if why_cues:
        roles.add("why")
    if len(why_cues) >= 2:
        roles.add("why+")                                 # a fuller statement of the suspected violation
    if ACTIVITY_RE.search(sent):
        roles.add("activity")
    if AMOUNT_RE.search(sent):
        roles.add("amount")
    if TOTAL_RE.search(sent):
        roles.add("total")
    if DATE_RE.search(sent):
        roles.add("when")
    if WHO_RE.search(sent) or any(len(n.split()) >= 2 for n in find_names(sent)):
        roles.add("who")                                  # a role word, or a multi-word name
    if WHERE_KW.search(sent) or WHERE_PLACE.search(sent):
        roles.add("where")
    if ACTION_RE.search(sent):
        roles.add("action")
    return roles


ROLE_WEIGHT = {"why": 2.5, "why+": 2.0, "total": 2.5, "amount": 1.0, "action": 2.5, "when": 1.5, "who": 1.5,
               "named": 2.0, "activity": 1.0, "where": 1.0}


# =============================================================================
# 3. Summarizers
# =============================================================================
def guided_summary(text, max_words=130):
    """Extractive: choose sentences that together cover why/what/amounts/when/who/where/action."""
    sents = split_sentences(text)
    n_words = len(text.split())
    if n_words <= 70:
        return " ".join(sents)                           # already short: nothing worth dropping
    budget = min(max_words, max(60, int(0.5 * n_words)))    # a summary should be clearly shorter
    roles = [sentence_roles(s) for s in sents]
    subjects = [n.lower() for n in top_names(text)]
    for i, s in enumerate(sents):
        if any(n in s.lower() for n in subjects):
            roles[i].add("named")                        # names one of the main subjects
    cent, sim = rank_sentences(sents)
    top = max(cent) or 1.0
    cent = [c / top for c in cent]
    wc = [len(s.split()) for s in sents]
    chosen, covered, used = [], set(), 0
    while True:
        best, best_val = None, 0.05
        for i in range(len(sents)):
            if i in chosen:
                continue
            if used + wc[i] > budget * (1.4 if not chosen else 1.0):
                continue
            if not (roles[i] - covered) and used >= 0.6 * budget:
                continue                                 # nothing new to say and the summary is long enough
            body = re.sub(r"^\(\d+\)\s*", "", sents[i])
            specifics = min(len(AMOUNT_RE.findall(body)) + len(DATE_RE.findall(body)) + len(find_names(body)), 6) / 6.0
            gain = sum(ROLE_WEIGHT[r] for r in roles[i] - covered) + 1.0 * cent[i] + 0.7 * specifics
            if i <= 2 and "why" in roles[i]:
                gain += 0.7                              # FinCEN: the introduction states the purpose of the SAR
            if chosen and max(sim[i][j] for j in chosen) > 0.8:
                gain *= 0.4                              # nearly a repeat of something already chosen
            if sents[i].startswith("(") and wc[i] < 14:
                gain *= 0.5                              # short list-item fragment
            if i > 0 and re.match(r"(He|She|They|It|His|Her|Their)\b", sents[i]):
                gain *= 0.6                              # pronoun with no antecedent in the summary
            val = gain / (max(wc[i], 3) ** 0.4)          # length penalty: favors substance over fragments
            if val > best_val:
                best, best_val = i, val
        if best is None:
            break
        chosen.append(best)
        covered |= roles[best]
        used += wc[best]
    return " ".join(re.sub(r"^\(\d+\)\s*", "", sents[i]) for i in sorted(chosen))   # drop "(2)" list markers


def bart_summary(text, max_words, model_name):
    """Abstractive (optional). Needs: pip install torch transformers"""
    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError:
        sys.exit("--engine bart needs:  pip install torch transformers")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).eval()
    limit = tok.model_max_length if tok.model_max_length and tok.model_max_length < 100_000 else 1024
    limit = min(limit, 1024)
    is_t5 = "t5" in model_name.lower()

    def gen(t):
        enc = tok(("summarize: " if is_t5 else "") + t, return_tensors="pt", truncation=True, max_length=limit)
        n = enc["input_ids"].shape[1]
        max_len = max(20, min(int(max_words * 1.4), n))
        min_len = max(10, min(int(max_words * 0.5), max_len // 2, n // 2))
        with torch.no_grad():
            out = model.generate(**enc, max_length=max_len, min_length=min_len, num_beams=4,
                                 no_repeat_ngram_size=3, length_penalty=2.0, early_stopping=True)
        return tok.decode(out[0], skip_special_tokens=True).strip()

    chunks, cur, cur_len = [], [], 0
    for s in split_sentences(text):
        n = len(tok.encode(s, add_special_tokens=False))
        if cur and cur_len + n > limit - 50:
            chunks.append(" ".join(cur))
            cur, cur_len = [], 0
        cur.append(s)
        cur_len += n
    if cur:
        chunks.append(" ".join(cur))
    parts = [gen(c) for c in chunks]
    return parts[0] if len(parts) == 1 else gen(" ".join(parts))


# =============================================================================
# 4. Checker: summary vs. narrative
# =============================================================================
NEGATIONS = {"not", "no", "never", "none", "neither", "nor", "cannot", "without", "nothing"}
OPPOSITES = [("increase", "decrease"), ("more", "less"), ("higher", "lower"), ("approve", "reject"),
             ("allow", "prohibit"), ("before", "after"), ("legal", "illegal"), ("valid", "invalid"),
             ("incoming", "outgoing"), ("deposit", "withdrawal"), ("open", "close"), ("sent", "received"),
             ("originator", "beneficiary"), ("accept", "refuse"), ("true", "false")]


def numbers(text):
    return {m.replace(",", "").rstrip(".") for m in re.findall(r"\d[\d,]*(?:\.\d+)?", text)}


def entities(sentence):
    """Capitalized words not at the start of the sentence, plus ALL-CAPS acronyms anywhere."""
    toks = re.findall(r"[A-Za-z][A-Za-z'-]*", sentence)
    out = []
    for i, t in enumerate(toks):
        if t.lower() in STOPWORDS:
            continue
        if t.isupper() and len(t) > 1:
            out.append(t)
        elif i > 0 and t[0].isupper():
            out.append(t)
    return out


def stem_to_word(text):
    m = {}
    for t in re.findall(r"[A-Za-z]+", text):
        if t.lower() not in STOPWORDS and len(t) > 1:
            m.setdefault(stem(t), t)
    return m


def best_window(stems, vocabs):
    """Best single narrative sentence or adjacent pair; on equal support the single sentence wins."""
    best_key, best = None, (0.0, (0, 0))
    n = len(vocabs)
    for i in range(n):
        for j in (i, i + 1):
            if j >= n:
                continue
            v = vocabs[i] if i == j else vocabs[i] | vocabs[j]
            score = sum(1 for s in stems if v.has(s)) / len(stems)
            key = (round(score, 6), -(j - i))
            if best_key is None or key > best_key:
                best_key, best = key, (score, (i, j))
    return best


def check_summary(source, summary, min_support=0.7):
    src_sents, sum_sents = split_sentences(source), split_sentences(summary)
    if not src_sents or not sum_sents:
        sys.exit("Narrative and summary must both contain text.")
    stem_lists = [content_stems(s) for s in src_sents]
    vocabs = [Vocab(l) for l in stem_lists]
    gvocab = Vocab([s for l in stem_lists for s in l])
    src_nums = numbers(source)
    src_tokens = set(re.findall(r"[A-Za-z][A-Za-z'-]*", source.lower()))

    results = []
    for k, sent in enumerate(sum_sents, 1):
        stems = content_stems(sent)
        issues, hard = [], False
        bad_nums = sorted(n for n in numbers(sent) if n not in src_nums)
        if bad_nums:
            issues.append(f"number(s) not in narrative: {', '.join(bad_nums)}")
            hard = True
        bad_ents = sorted({e for e in entities(sent) if e.lower() not in src_tokens})
        if bad_ents:
            issues.append(f"name(s) not in narrative: {', '.join(bad_ents)}")
            hard = True
        support, (i, j) = best_window(stems, vocabs) if stems else (1.0, (0, 0))
        novel = sorted({s for s in stems if not gvocab.has(s)})
        window_text = " ".join(src_sents[i: j + 1]).lower()
        wvocab = Vocab(content_stems(window_text))
        neg_flag = bool(NEGATIONS & set(re.findall(r"[a-z]+", sent.lower()))) != \
            bool(NEGATIONS & set(re.findall(r"[a-z]+", window_text)))
        if neg_flag:
            issues.append("negation differs from the closest narrative passage")
        sset = set(stems)
        reversal = False
        for a, b in OPPOSITES:
            for x, y in ((a, b), (b, a)):
                if stem(x) in sset and not wvocab.has(stem(x)) and wvocab.has(stem(y)):
                    issues.append(f"possible reversal: summary says '{x}', narrative says '{y}'")
                    reversal = True
        if novel:
            shown = stem_to_word(sent)
            issues.append(f"wording not found in narrative: {', '.join(shown.get(s, s) for s in novel[:6])}")
        if support < 0.5:
            verdict = "UNSUPPORTED"
            issues.insert(0, f"only {support:.0%} of its key words appear near any narrative passage")
        elif hard:
            verdict = "UNSUPPORTED"
        elif neg_flag or reversal or support < min_support or len(novel) >= 2:
            verdict = "REVIEW"
        else:
            verdict = "OK"
        results.append({"n": k, "text": sent, "verdict": verdict, "support": support,
                        "window": (i + 1, j + 1), "issues": issues, "hard": hard})

    notes = []
    words = re.findall(r"[a-z']+", summary.lower())
    tri = [tuple(words[i: i + 3]) for i in range(len(words) - 2)]
    rep = sum(c - 1 for c in Counter(tri).values() if c > 1) / len(tri) if tri else 0.0
    if rep > 0.15:
        notes.append(f"repetition: {rep:.0%} of word triples repeat")
    if len(source.split()) > 120 and len(summary.split()) > 0.6 * len(source.split()):
        notes.append("summary is more than 60% of the narrative length (barely shorter)")

    return {"results": results, "notes": notes, "src_sents": src_sents, "kept": retained(source, summary)}


def retained(source, summary):
    """Did the summary keep the facts a SAR reader needs? Returns [(label, ok_or_None, detail)]."""
    out = []
    top = top_names(source)
    if top:
        hit = [n for n in top if n.lower() in summary.lower()]
        out.append(("subject", bool(hit), hit[0] if hit else f"none of {', '.join(top)}"))
    keys = key_amounts(source)
    if keys:
        have = [v for v in keys if any(abs(v - a) < 1e-6 for a in amounts(summary))]
        out.append(("amount (key)", bool(have),
                    fmt_money(have[0]) + " kept" if have else "none of " + ", ".join(fmt_money(v) for v in keys[:3])))
    if DATE_RE.search(source):
        out.append(("period/dates", bool(DATE_RE.search(summary)), ""))
    typ = typologies(source)
    if typ:
        t2 = set(typologies(summary))
        out.append(("activity", bool(t2), ", ".join(sorted(t2)[:3]) if t2 else "none detected"))
    if ACTION_RE.search(source):
        out.append(("follow-up action", bool(ACTION_RE.search(summary)), ""))
    if WHY_RE.search(source):
        out.append(("why suspicious", bool(WHY_RE.search(summary)), ""))
    return out


def overall_verdict(res):
    if any(r["verdict"] == "UNSUPPORTED" for r in res["results"]):
        return "FAIL"
    if any(r["verdict"] == "REVIEW" for r in res["results"]) or res["notes"] or \
            any(ok is False for _, ok, _ in res["kept"]):
        return "REVIEW"
    return "PASS"


class NLI:
    """Optional PyTorch entailment model (--nli). pip install torch transformers"""

    def __init__(self, name):
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError:
            sys.exit("--nli needs:  pip install torch transformers")
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).eval()
        self.idx = {}
        for i, lab in self.model.config.id2label.items():
            lab = str(lab).lower()
            for key, frag in (("entail", "entail"), ("contra", "contra"), ("neutral", "neutral")):
                if frag in lab:
                    self.idx[key] = int(i)
        if len(self.idx) != 3:
            sys.exit(f"{name} does not look like an NLI model.")

    def probs(self, premise, hypothesis):
        enc = self.tok(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512)
        with self.torch.no_grad():
            p = self.torch.softmax(self.model(**enc).logits[0], dim=-1).tolist()
        return {k: p[i] for k, i in self.idx.items()}


def apply_nli(result, nli, top_k=4):
    """Let the entailment model decide meaning-level support; hard number/name findings still stand."""
    src = result["src_sents"]
    vocabs = [Vocab(content_stems(s)) for s in src]
    for r in result["results"]:
        stems = content_stems(r["text"])
        cands = []
        for i in range(len(src)):
            for j in (i, i + 1):
                if j < len(src):
                    v = vocabs[i] if i == j else vocabs[i] | vocabs[j]
                    sc = sum(1 for s in stems if v.has(s)) / len(stems) if stems else 0.0
                    cands.append((-sc, j - i, i, j))
        cands.sort()
        best_e, win, max_c = 0.0, None, 0.0
        for _, _, i, j in cands[:top_k]:
            p = nli.probs(" ".join(src[i: j + 1]), r["text"])
            if win is None or p["entail"] > best_e:
                best_e, win = p["entail"], (i + 1, j + 1)
            max_c = max(max_c, p["contra"])
        keep = [x for x in r["issues"] if x.startswith(("number(s)", "name(s)"))]
        if r["hard"]:
            verdict = "UNSUPPORTED"
        elif best_e >= 0.5:
            verdict = "OK"
        elif max_c >= 0.5 and best_e < 0.3:
            verdict, keep = "UNSUPPORTED", keep + [f"contradicted by the narrative ({max_c:.0%})"]
        elif best_e < 0.2:
            verdict, keep = "UNSUPPORTED", keep + ["not backed up by any narrative passage"]
        else:
            verdict, keep = "REVIEW", keep + ["only partly backed up by the narrative"]
        r.update(verdict=verdict, issues=keep, window=win, support=best_e)


# =============================================================================
# 5. Narrative completeness (FinCEN guide: who/what/when/where/why/how + follow-up)
# =============================================================================
def completeness(text):
    names = find_names(text)
    who = bool(names) or len({m.lower() for m in WHO_RE.findall(text)}) >= 2
    what = bool(typologies(text)) and (bool(amounts(text)) or len(ACTIVITY_RE.findall(text)) >= 2)
    items = [
        ("who", who), ("what", what), ("when", bool(DATE_RE.search(text))),
        ("where", bool(WHERE_KW.search(text) or WHERE_PLACE.search(text))),
        ("why", bool(WHY_RE.search(text))),
        ("how", bool(HOW_RE.search(text)) and len(ACTIVITY_RE.findall(text)) >= 2),
        ("follow-up action", bool(ACTION_RE.search(text))),
        ("records / case ref", bool(RECORDS_RE.search(text))),
    ]
    score = sum(ok for _, ok in items)
    label = "strong" if score >= 7 else "adequate" if score >= 5 else "thin"
    warns = []
    if ATTACH_RE.search(text):
        warns.append("refers to an attachment: attachments are not entered into the SAR system; describe it in the narrative")
    if len(re.findall(r"\bN\.?A\.?\b", text)) >= 3:
        warns.append("checklist-style answers (N.A.) instead of a chronological narrative")
    if len(text.split()) < 40:
        warns.append(f"very short ({len(text.split())} words)")
    if not amounts(text):
        warns.append("no dollar amounts given")
    return {"items": items, "score": score, "label": label, "warnings": warns}


# =============================================================================
# 6. Reports and CLI
# =============================================================================
def short(s, n=100):
    return s if len(s) <= n else s[: n - 3].rstrip() + "..."


def run_one(text, args, summary_text=None):
    text = clean(text)
    if summary_text is not None:
        summary = clean(summary_text)
    elif args.engine == "bart":
        summary = bart_summary(text, args.words, args.model)
    else:
        summary = guided_summary(text, args.words)
    summary = re.sub(r"\s+", " ", summary).strip()          # one paragraph
    res = check_summary(text, summary)
    if args.nli:
        apply_nli(res, NLI(args.nli_model))
    res["overall"] = overall_verdict(res)
    res["summary"] = summary
    res["narr_words"] = len(text.split())
    res["sum_words"] = len(summary.split())
    res["completeness"] = completeness(text)
    res["typologies"] = typologies(text)
    return res


def print_report(name, r):
    c = r["completeness"]
    print(f"SAR NARRATIVE: {name}  ({r['narr_words']} words)")
    print()
    print(f"SUMMARY ({r['sum_words']} words)")
    print(textwrap.fill(r["summary"], 100, initial_indent="  ", subsequent_indent="  "))
    print()
    print(f"CHECK: {r['overall']}   (summary vs. narrative)")
    for x in r["results"]:
        a, b = x["window"]
        where = f"narrative sentence {a}" if a == b else f"narrative sentences {a}-{b}"
        print(f"  [{x['n']}] {x['verdict']:<11} support {x['support']:.0%} ({where})  \"{short(x['text'], 60)}\"")
        for issue in x["issues"]:
            print(f"        - {issue}")
    for n in r["notes"]:
        print(f"  NOTE: {n}")
    kept = "; ".join(f"{lab} {'yes' if ok else 'NO'}" + (f" ({d})" if d and lab != 'activity' else "")
                     for lab, ok, d in r["kept"])
    print(f"  Summary keeps: {kept}")
    print()
    print(f"NARRATIVE COMPLETENESS vs. FinCEN guide: {c['score']}/8 ({c['label']})")
    print("  " + "  ".join(f"{lab}: {'yes' if ok else 'NO'}" for lab, ok in c["items"]))
    for w in c["warnings"]:
        print(f"  WARNING: {w}")
    if r["typologies"]:
        print(f"  Activity detected: {', '.join(r['typologies'])}")


def main():
    ap = argparse.ArgumentParser(description="Summarize a SAR narrative into one paragraph, then check it.")
    ap.add_argument("narrative", nargs="?", help="narrative text file, or - for stdin")
    ap.add_argument("--batch", metavar="DIR", help="process every .txt file in a folder")
    ap.add_argument("--summary", metavar="FILE", help="check this existing summary instead of generating one")
    ap.add_argument("--words", type=int, default=130, help="target summary length in words (default 130)")
    ap.add_argument("--engine", choices=["guided", "bart"], default="guided", help="summarizer (default guided)")
    ap.add_argument("--model", default="sshleifer/distilbart-cnn-12-6", help="model for --engine bart")
    ap.add_argument("--nli", action="store_true", help="add a PyTorch entailment check (needs torch, transformers)")
    ap.add_argument("--nli-model", default="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
    ap.add_argument("-v", "--verbose", action="store_true", help="in batch mode, print each summary")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

    rank = {"PASS": 0, "REVIEW": 1, "FAIL": 2}
    if args.batch:
        files = sorted(Path(args.batch).glob("*.txt"))
        if not files:
            sys.exit(f"No .txt files in {args.batch}")
        print(f"{'file':<22}{'words':>6}{'summ':>6}  {'check':<7}{'complete':<14}summary keeps")
        worst = 0
        for f in files:
            r = run_one(f.read_text(encoding="utf-8"), args)
            worst = max(worst, rank[r["overall"]])
            c = r["completeness"]
            keeps = " ".join(f"{lab.split()[0]}={'Y' if ok else 'N'}" for lab, ok, _ in r["kept"])
            print(f"{f.name:<22}{r['narr_words']:>6}{r['sum_words']:>6}  {r['overall']:<7}"
                  f"{c['score']}/8 {c['label']:<9}{keeps}")
            if args.verbose:
                print(textwrap.fill(r["summary"], 100, initial_indent="    ", subsequent_indent="    "))
                print()
        sys.exit(worst)

    if not args.narrative:
        ap.error("give a narrative file (or - for stdin), or use --batch DIR")
    text = sys.stdin.read() if args.narrative == "-" else Path(args.narrative).read_text(encoding="utf-8")
    if not text.strip():
        sys.exit("No text to summarize.")
    summary_text = Path(args.summary).read_text(encoding="utf-8") if args.summary else None
    r = run_one(text, args, summary_text)
    print_report("stdin" if args.narrative == "-" else Path(args.narrative).name, r)
    sys.exit(rank[r["overall"]])


if __name__ == "__main__":
    main()

"""Text processing for agent memory: tokens for BM25, features for learning.

Everything here is stdlib-only and deterministic.

Two different things are extracted from a memory's text:

* **Index terms** (:func:`terms`) feed the lexical (BM25) index: lowercased
  word tokens, English stop words removed, a light suffix-stripping stemmer
  applied. They are *not* HeapDB features: thousands of distinct words would
  make the noisy-OR usefulness term meaningless (it saturates) and would copy
  memory content into the link table.
* **Shape features** (:func:`memory_features`) describe *what kind of*
  memory this is, with a fixed set of slots that every memory fills exactly
  once (``kind:``, ``source:``, ``len:``, ``date:``, ``num:``, ``q:``,
  ``ent:``, ``topic:``) plus optional user ``tag:`` features. These are the
  HeapDB event features whose links to the effect ``useful`` are learned
  from feedback ("preferences are often used", "tool output rarely is").
  Because the slot count is fixed, memories are comparable under noisy-OR.

:func:`query_type` classifies a query (``when``, ``who``, ...) and
:func:`pair_features` crosses it with a memory's shape so usefulness can be
learned *per kind of question* (e.g. ``qt:when&date:yes``).
"""

from __future__ import annotations

import re
from typing import Iterable, Iterator, Sequence

__all__ = [
    "STOPWORDS",
    "TOPICS",
    "content_terms",
    "detect_topic",
    "memory_features",
    "pair_features",
    "query_type",
    "stem",
    "terms",
    "tokenize",
]

_WORD = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?")

#: A compact English stop-word list (articles, pronouns, auxiliaries,
#: prepositions, conjunctions and conversational fillers).
STOPWORDS = frozenset("""
a about above after again against all am an and any are aren't as at be because been before
being below between both but by can can't cannot could couldn't did didn't do does doesn't
doing don't down during each few for from further had hadn't has hasn't have haven't having
he he'd he'll he's her here here's hers herself him himself his how how's i i'd i'll i'm i've
if in into is isn't it it's its itself let's me more most mustn't my myself no nor not of off
on once only or other ought our ours ourselves out over own same shan't she she'd she'll she's
should shouldn't so some such than that that's the their theirs them themselves then there
there's these they they'd they'll they're they've this those through to too under until up
very was wasn't we we'd we'll we're we've were weren't what what's when when's where where's
which while who who's whom why why's with won't would wouldn't you you'd you'll you're you've
your yours yourself yourselves also just really yeah yes oh ok okay hey hi hello thanks thank
s t d ll m re ve got get gets getting will
""".split())

_MONTHS = ("january february march april may june july august september october november "
           "december jan feb mar apr jun jul aug sep sept oct nov dec").split()
_WEEKDAYS = "monday tuesday wednesday thursday friday saturday sunday".split()
_TIME_WORDS = set(_MONTHS) | set(_WEEKDAYS) | set(
    "yesterday today tomorrow tonight ago week weeks weekend month months year years day days "
    "morning evening afternoon night last next recently soon since until summer winter spring "
    "autumn fall birthday anniversary holiday christmas date dates".split())
_NUMBER_WORDS = set(
    "one two three four five six seven eight nine ten eleven twelve twenty thirty forty fifty "
    "hundred thousand million first second third fourth fifth half dozen couple".split())
_YEAR = re.compile(r"\b(19|20)\d{2}\b")
_DIGIT = re.compile(r"\d")
_CAP = re.compile(r"(?<![.!?]\s)(?<!^)\b[A-Z][a-z]{2,}")

_TOPIC_WORDS: dict[str, str] = {
    "work": "work job career office boss colleague project meeting deadline client company "
            "promotion salary internship business team manager hired",
    "family": "family kids child children son daughter mom dad mother father parents husband "
              "wife sister brother grandma grandpa baby partner married",
    "health": "health doctor sick illness hospital medicine therapy pain allergy diet exercise "
              "gym running yoga sleep stress anxiety mental injury",
    "food": "food eat cooking recipe dinner lunch breakfast restaurant meal coffee tea vegan "
            "vegetarian baking pizza drink",
    "travel": "travel trip vacation flight hotel visit country city beach road camping hike "
              "tour abroad journey",
    "hobby": "hobby painting music guitar violin piano book reading art game sport photography "
             "dance singing garden craft pottery writing",
    "money": "money pay buy price cost budget saving bank loan rent invest dollars afford purchase",
    "education": "school college university class course study learning degree exam teacher "
                 "student graduate education",
    "tech": "code bug deploy server api database python javascript software app computer test "
            "repo error config",
    "preference": "like love prefer favorite favourite hate dislike enjoy want wish always never",
}


def tokenize(text: str) -> list[str]:
    """Lowercased word tokens (letters/digits, inner apostrophes kept)."""
    return [m.group(0).lower() for m in _WORD.finditer(text or "")]


_SUFFIXES = ("ational", "tional", "ization", "fulness", "ousness", "iveness", "ements", "ement",
             "ments", "ment", "ingly", "edly", "ings", "ing", "ies", "ied", "ers", "er",
             "ed", "es", "ly", "s")


def stem(word: str) -> str:
    """A light suffix-stripping stemmer (not Porter, but in its spirit).

    Strips one common English suffix if at least three characters remain,
    maps ``-ies``/``-ied`` to ``-i``, undoubles a final consonant after
    ``-ing``/``-ed`` (``running`` -> ``run``) and maps a trailing ``y`` to
    ``i`` so ``study``/``studies``/``studied`` share a stem.
    """
    w = word.lower()
    if len(w) <= 3 or w.isdigit():
        return w
    if w.endswith("'s"):
        w = w[:-2]
    for suf in _SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            if suf in ("ies", "ied"):
                return w[: -len(suf)] + "i"
            if suf == "s" and w.endswith("ss"):
                break
            if suf == "es" and not re.search(r"(ss|x|ch|sh|z)es$", w):
                w = w[:-1]  # "notes" -> "note"; "boxes" -> "box" below
                break
            w = w[: -len(suf)]
            if suf in ("ing", "ed", "er", "ers", "ings", "edly", "ingly") and len(w) >= 4 \
                    and w[-1] == w[-2] and w[-1] not in "lsz":
                w = w[:-1]
            break
    if w.endswith("y") and len(w) > 3:
        w = w[:-1] + "i"
    if w.endswith("e") and len(w) >= 4:
        w = w[:-1]
    return w


#: Small closed topic vocabulary: topic -> trigger stems (built by stemming
#: the trigger words above, so triggers always match :func:`stem`). One
#: ``topic:`` feature per memory (best match or ``topic:none``) keeps the
#: feature count fixed. Pass ``topics=`` to :func:`memory_features` to extend.
TOPICS: dict[str, frozenset[str]] = {
    name: frozenset(stem(w) for w in words.split()) for name, words in _TOPIC_WORDS.items()
}


def terms(text: str, stopwords: frozenset[str] = STOPWORDS) -> list[str]:
    """Index terms for BM25: tokens minus stop words, stemmed. Keeps order
    and duplicates (term frequency matters)."""
    out = []
    for t in tokenize(text):
        if t in stopwords or (len(t) == 1 and not t.isdigit()):
            continue
        out.append(stem(t))
    return out


def content_terms(text: str) -> set[str]:
    """Distinct index terms (used by the importance heuristic)."""
    return set(terms(text))


def detect_topic(stems: Iterable[str], topics: dict[str, frozenset[str]] = TOPICS) -> str:
    """Best-matching topic name by trigger count, or ``"none"``."""
    s = set(stems)
    best, best_n = "none", 0
    for name in sorted(topics):
        n = len(s & topics[name])
        if n > best_n:
            best, best_n = name, n
    return best


def _len_bucket(n_terms: int) -> str:
    if n_terms < 5:
        return "short"
    if n_terms < 15:
        return "medium"
    return "long"


def _has_date(tokens: Sequence[str], text: str) -> bool:
    return bool(_YEAR.search(text)) or any(t in _TIME_WORDS for t in tokens)


def _has_number(tokens: Sequence[str], text: str) -> bool:
    return bool(_DIGIT.search(text)) or any(t in _NUMBER_WORDS for t in tokens)


def _has_entity(text: str) -> bool:
    # A capitalised word that does not start a sentence: a cheap proxy for
    # "names a person, place, product or organisation".
    for sent in re.split(r"(?<=[.!?])\s+", text.strip()):
        words = sent.split()
        if any(w[:1].isupper() and w[1:2].islower() and len(w.strip(".,!?;:'\"")) > 2
               for w in words[1:]):
            return True
    return False


def memory_features(
    text: str,
    kind: str,
    source: str | None = None,
    tags: Iterable[str] = (),
    topics: dict[str, frozenset[str]] = TOPICS,
) -> frozenset[str]:
    """The HeapDB features of a memory: one value per fixed slot, plus tags.

    ``kind:<kind>``, ``source:<source|unknown>``, ``len:short|medium|long``
    (content terms), ``date:yes|no`` (temporal expression), ``num:yes|no``,
    ``q:yes|no`` (the memory is itself a question), ``ent:yes|no`` (names a
    proper noun), ``topic:<name>|none``, and ``tag:<t>`` per user tag.
    """
    toks = tokenize(text)
    stems = [stem(t) for t in toks if t not in STOPWORDS]
    feats = {
        f"kind:{kind}",
        f"source:{(source or 'unknown').strip().lower().replace(' ', '_')}",
        f"len:{_len_bucket(len(stems))}",
        f"date:{'yes' if _has_date(toks, text) else 'no'}",
        f"num:{'yes' if _has_number(toks, text) else 'no'}",
        f"q:{'yes' if text.strip().endswith('?') else 'no'}",
        f"ent:{'yes' if _has_entity(text) else 'no'}",
        f"topic:{detect_topic(stems, topics)}",
    }
    feats.update(f"tag:{t.strip().lower()}" for t in tags if t and t.strip())
    return frozenset(feats)


_QTYPES = ("when", "who", "where", "why", "how", "which", "what")
_YESNO = ("is", "are", "was", "were", "do", "does", "did", "can", "could", "would", "will",
          "should", "has", "have", "had")


def query_type(query: str) -> str:
    """Coarse query type: the first wh-word, ``yesno`` for auxiliary-initial
    questions, ``hypothetical`` for "would/might/likely" questions, else
    ``other``. ``how many/much/long/old`` map to ``howmany`` and questions
    asking for a year/date map to ``when``."""
    toks = tokenize(query)
    if not toks:
        return "other"
    s = set(toks)
    if "likely" in s or "might" in s or (toks[0] == "would"):
        return "hypothetical"
    for i, t in enumerate(toks):
        if t in _QTYPES:
            if t == "how" and i + 1 < len(toks) and toks[i + 1] in ("many", "much", "long", "old", "often"):
                return "howmany"
            if t in ("what", "which") and i + 1 < len(toks) and toks[i + 1] in (
                    "year", "month", "day", "date", "time", "week"):
                return "when"
            return t
    if toks[0] in _YESNO:
        return "yesno"
    return "other"


#: Shape slots crossed with the query type in :func:`pair_features`.
CROSS_SLOTS = ("date", "num", "len", "q", "ent")


def pair_features(query: str | None, features: Iterable[str]) -> frozenset[str]:
    """Features of one *retrieval* (query x memory) for usefulness learning.

    The memory's own features plus, if a query is given, ``qt:<type>&<slot>``
    for each shape slot in :data:`CROSS_SLOTS` (e.g. ``qt:when&date:yes``),
    so that "date-bearing memories are useful for *when* questions" can be
    learned separately from "date-bearing memories are useful"; and
    ``qsrc:yes|no``: does the query mention the memory's source (e.g. the
    person who said it)?
    """
    feats = set(features)
    if query:
        qt = query_type(query)
        qtoks = set(tokenize(query))
        for f in list(feats):
            slot, _, val = f.partition(":")
            if slot in CROSS_SLOTS:
                feats.add(f"qt:{qt}&{f}")
            elif slot == "source" and val != "unknown":
                feats.add(f"qsrc:{'yes' if set(val.split('_')) & qtoks else 'no'}")
    return frozenset(feats)


def iter_ngrams(tokens: Sequence[str], n: int) -> Iterator[str]:
    for i in range(len(tokens) - n + 1):
        yield " ".join(tokens[i: i + n])

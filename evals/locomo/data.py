"""LoCoMo loading: download at run time (CC BY-NC 4.0, not redistributed here),
parse sessions, timestamps, QA items and evidence ids.

LoCoMo: Maharana et al., "Evaluating Very Long-Term Conversational Memory of
LLM Agents", ACL 2024 (arXiv:2402.17753). https://github.com/snap-research/locomo
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
#: sha256 of the file this harness was developed and evaluated against.
SHA256 = "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"

#: LoCoMo's numeric QA categories. The paper lists the types but the data
#: file only has codes; this mapping is read off the questions (and matches
#: the one commonly used by memory-system evaluations on LoCoMo).
CATEGORIES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}

_EVID = re.compile(r"D:?(\d+):(\d+)")
_SESSION = re.compile(r"session_(\d+)")


@dataclass
class Turn:
    dia_id: str
    speaker: str
    text: str          # what is indexed: "Speaker: utterance [shares <caption>]"
    time: float        # session start + 60 s per turn (keeps order)
    session: int


@dataclass
class QA:
    question: str
    category: int
    evidence: frozenset[str]
    raw_evidence: tuple[str, ...]


@dataclass
class Conversation:
    sample_id: str
    turns: list[Turn]
    qa: list[QA]
    now: float                      # query time: last session + 1 day
    excluded: dict[str, int] = field(default_factory=dict)


def cache_dir() -> Path:
    base = os.environ.get("HEAPDB_MEMORY_CACHE") or os.path.join(
        os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "heapdb-memory")
    return Path(base)


def download(dest: Path | None = None, url: str = URL) -> Path:
    """Fetch locomo10.json into the cache (once) and return its path."""
    dest = dest or cache_dir() / "locomo" / "locomo10.json"
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".part")
        with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as fh:
            fh.write(r.read())
        tmp.replace(dest)
    return dest


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_time(s: str) -> float:
    dt = datetime.strptime(" ".join(s.split()), "%I:%M %p on %d %B, %Y")
    return dt.replace(tzinfo=timezone.utc).timestamp()


def parse_evidence(raw: list[str]) -> list[str]:
    """Normalise evidence ids ("D8:6; D9:17", "D:11:26", "D30:05" occur)."""
    out = []
    for e in raw:
        for s, t in _EVID.findall(str(e)):
            out.append(f"D{int(s)}:{int(t)}")
    return out


def turn_text(t: dict) -> str:
    text = f"{t['speaker']}: {t.get('text', '')}".strip()
    if t.get("blip_caption"):
        text += f" [shares {t['blip_caption']}]"
    return text


def load(path: Path) -> list[Conversation]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    convs = []
    for x in data:
        c = x["conversation"]
        sessions = sorted((int(m.group(1)) for k in c if (m := _SESSION.fullmatch(k))))
        turns: list[Turn] = []
        last = 0.0
        for s in sessions:
            t0 = parse_time(c[f"session_{s}_date_time"])
            for i, t in enumerate(c[f"session_{s}"]):
                turns.append(Turn(t["dia_id"], t["speaker"], turn_text(t), t0 + 60.0 * i, s))
            last = max(last, t0 + 60.0 * len(c[f"session_{s}"]))
        ids = {t.dia_id for t in turns}
        qa, excl = [], {"no_evidence": 0, "evidence_not_found": 0, "evidence_partly_dropped": 0}
        for q in x["qa"]:
            raw = q.get("evidence") or []
            ev = parse_evidence(raw)
            valid = frozenset(e for e in ev if e in ids)
            if not raw:
                excl["no_evidence"] += 1
                continue
            if not valid:
                excl["evidence_not_found"] += 1
                continue
            if len(valid) < len(set(ev)):
                excl["evidence_partly_dropped"] += 1
            qa.append(QA(q["question"], int(q["category"]), valid, tuple(map(str, raw))))
        convs.append(Conversation(x["sample_id"], turns, qa, last + 86400.0, excl))
    return convs

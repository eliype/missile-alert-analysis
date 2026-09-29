"""Transparent rules for turning official IDF posts into incoming threats."""

from __future__ import annotations

import re
from datetime import datetime, timedelta

import pandas as pd


IRAN_PATTERNS = [
    re.compile(r"\b(?:missiles?|projectiles?|rockets?|uavs?|unmanned aerial vehicles?|aerial targets?)\b.{0,55}\b(?:launched|crossed) from iran\b", re.I),
    re.compile(r"\blaunch(?:es|ed)? from iran\b", re.I),
    re.compile(r"\biran(?:ian)?\b.{0,35}\b(?:launched|fired)\b.{0,45}\b(?:toward|at)\b.{0,25}\bisrael", re.I),
]

LEBANON_PATTERNS = [
    re.compile(r"\b(?:missiles?|projectiles?|rockets?|uavs?|unmanned aerial vehicles?|aerial targets?)\b.{0,55}\b(?:launched|crossed) from (?:lebanon|lebanese territory)\b", re.I),
    re.compile(r"\blaunch(?:es|ed)? from (?:lebanon|lebanese territory)\b", re.I),
    re.compile(r"\bcrossed from lebanon\b", re.I),
    re.compile(r"\bhezbollah\b.{0,60}\b(?:launched|fired)\b.{0,55}\b(?:toward|at)\b.{0,30}\b(?:israel|israeli territory|state of israel)\b", re.I),
]

OTHER_SOURCE_PATTERNS = {
    "yemen": [
        re.compile(r"\b(?:launched|crossed) from yemen\b", re.I),
        re.compile(r"\blaunch(?:es|ed)? from yemen\b", re.I),
    ],
    "gaza": [
        re.compile(r"\b(?:launched|crossed) from (?:the )?gaza strip\b", re.I),
        re.compile(r"\blaunch(?:es|ed)? from gaza\b", re.I),
    ],
    "syria": [
        re.compile(r"\b(?:launched|crossed) from syria\b", re.I),
        re.compile(r"\blaunch(?:es|ed)? from syria\b", re.I),
    ],
}

EXPLICIT_SIREN_TIME = re.compile(
    r"\bsirens?\b.{0,100}?\b(?:at|around)\s+(\d{1,2}:\d{2})\b", re.I
)


def _matches_any(text: str, patterns: list[re.Pattern[str]]) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def classify_source(text: str) -> tuple[str, str]:
    iran = _matches_any(text, IRAN_PATTERNS)
    lebanon = _matches_any(text, LEBANON_PATTERNS)
    if iran and lebanon:
        return "mixed", "high"
    if iran:
        return "iran", "high"
    if lebanon:
        return "lebanon", "high"
    for source, patterns in OTHER_SOURCE_PATTERNS.items():
        if _matches_any(text, patterns):
            return source, "high"
    return "unknown", "none"


def classify_threat(text: str) -> str:
    lower = text.lower()
    found = []
    if re.search(r"\b(?:uav|uavs|unmanned aerial|aerial target|hostile aircraft)\b", lower):
        found.append("uav")
    if re.search(r"\bmissiles?\b", lower):
        found.append("missile")
    if re.search(r"\brockets?\b", lower):
        found.append("rocket")
    if re.search(r"\bprojectiles?\b", lower):
        found.append("projectile")
    return found[0] if len(found) == 1 else ("mixed" if found else "unknown")


def classify_time_basis(text: str) -> str:
    lower = text.lower()
    if EXPLICIT_SIREN_TIME.search(text):
        return "explicit_siren_time"
    if "following the sirens" in lower or "following sirens" in lower:
        return "confirmation_after_alert"
    if (
        "identified missiles launched" in lower
        or "identified missile launches" in lower
        or "identified a missile launched" in lower
    ):
        return "detection_before_alert"
    if "sirens" in lower and "short while ago" in lower:
        return "confirmation_after_alert"
    return "launch_report"


def is_relevant_incoming_threat(text: str, source: str) -> bool:
    if source not in {"iran", "lebanon", "mixed", "yemen", "gaza", "syria"}:
        return False
    lower = text.lower()
    stale_context = [
        "yesterday",
        "earlier today",
        "overnight",
        "last night",
        "since the beginning",
        "since the start",
        "throughout the operation",
    ]
    # Retrospective summaries are useful context but their post time is not an
    # event time. Keep them only when the post supplies an explicit siren time.
    if any(marker in lower for marker in stale_context) and not EXPLICIT_SIREN_TIME.search(text):
        return False
    if "sirens" not in lower:
        non_event_operations = re.search(
            r"\b(?:the )?idf (?:has )?(?:completed|conducted|struck|eliminated|dismantled|began)\b",
            lower,
        )
        prevented_launch = "prevented the launch" in lower or "thwart" in lower
        if non_event_operations or prevented_launch:
            return False
    if "no sirens were sounded" in lower and "sirens" not in lower.replace(
        "no sirens were sounded", ""
    ):
        return False
    evidence_terms = [
        "sirens",
        "into israeli territory",
        "toward israeli territory",
        "toward the territory of the state of israel",
        "toward the state of israel",
        "toward israel",
        "identified missiles launched",
        "intercepted",
        "impact",
    ]
    return any(term in lower for term in evidence_terms)


def referenced_siren_times(text: str, post_time_local: datetime) -> list[datetime]:
    results = []
    for match in EXPLICIT_SIREN_TIME.finditer(text):
        hour, minute = (int(part) for part in match.group(1).split(":"))
        if hour > 23 or minute > 59:
            continue
        same_day = post_time_local.replace(hour=hour, minute=minute, second=0, microsecond=0)
        candidates = [same_day - timedelta(days=1), same_day, same_day + timedelta(days=1)]
        # Official follow-up posts normally reference the nearest earlier time.
        earlier = [candidate for candidate in candidates if candidate <= post_time_local + timedelta(minutes=5)]
        chosen = min(
            earlier or candidates,
            key=lambda candidate: abs((post_time_local - candidate).total_seconds()),
        )
        results.append(chosen)
    return sorted(set(results))


def extract_events(messages: pd.DataFrame) -> pd.DataFrame:
    """Return one event per official post or explicitly referenced siren time."""
    records = []
    for row in messages.itertuples(index=False):
        text = "" if pd.isna(row.text) else str(row.text)
        source, source_confidence = classify_source(text)
        if not is_relevant_incoming_threat(text, source):
            continue
        threat = classify_threat(text)
        post_time = datetime.fromisoformat(row.timestamp_local)
        basis = classify_time_basis(text)
        references = referenced_siren_times(text, post_time)
        event_times = references if references else [post_time]
        if references:
            basis = "explicit_siren_time"

        for sequence, event_time in enumerate(event_times, start=1):
            event_id = f"tg_{int(row.post_id)}_{sequence}"
            eligible = (
                source in {"iran", "lebanon"}
                and threat != "uav"
                and source_confidence == "high"
            )
            records.append(
                {
                    "event_id": event_id,
                    "post_id": int(row.post_id),
                    "post_time_local": post_time.isoformat(),
                    "event_time_local": event_time.isoformat(),
                    "time_basis": basis,
                    "reported_source": source,
                    "source_confidence": source_confidence,
                    "threat_type": threat,
                    "eligible_for_missile_wave_match": eligible,
                    "text": text,
                    "source_url": row.source_url,
                }
            )
    return pd.DataFrame(records)

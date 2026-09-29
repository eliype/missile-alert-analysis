"""Reconstruct alert waves and match official source reports using time only."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def reconstruct_missile_waves(
    alerts_path: Path, gap_minutes: int = 8
) -> tuple[pd.DataFrame, pd.DataFrame]:
    alerts = pd.read_csv(alerts_path)
    missile = alerts.loc[alerts["category"] == 1].copy()
    missile["timestamp"] = pd.to_datetime(missile["alertDate"], errors="raise")
    missile = (
        missile.drop_duplicates(["alertDate", "data"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    missile["new_wave"] = (
        missile["timestamp"]
        .diff()
        .dt.total_seconds()
        .gt(gap_minutes * 60)
        .fillna(True)
    )
    missile["wave_id"] = missile["new_wave"].cumsum().astype(int)
    waves = missile.groupby("wave_id", as_index=False).agg(
        wave_start=("timestamp", "min"),
        wave_end=("timestamp", "max"),
        alert_rows=("data", "size"),
        zone_count=("data", "nunique"),
        burst_count=("timestamp", "nunique"),
    )
    waves["duration_minutes"] = (
        waves["wave_end"] - waves["wave_start"]
    ).dt.total_seconds() / 60
    waves["wave_family_proxy"] = np.where(
        waves["zone_count"] >= 26, "broad", "localized"
    )
    return missile, waves


def make_wave_localities(
    missile_rows: pd.DataFrame, labeled_waves: pd.DataFrame
) -> pd.DataFrame:
    """Create one auditable row for every locality alerted in every wave.

    The raw alert table is locality-level, so this preserves the geographical
    footprint that is summarized by ``zone_count`` in the wave table.  Wave
    labels are joined only after the time-based source matching is complete.
    """
    columns = [
        "wave_id",
        "locality",
        "first_alert_time",
        "last_alert_time",
        "alert_count",
        "wave_start",
        "wave_end",
        "zone_count",
        "wave_family_proxy",
        "reported_source",
        "source_label_confidence",
    ]
    if missile_rows.empty:
        return pd.DataFrame(columns=columns)

    locality_rows = (
        missile_rows.groupby(["wave_id", "data"], as_index=False)
        .agg(
            first_alert_time=("timestamp", "min"),
            last_alert_time=("timestamp", "max"),
            alert_count=("timestamp", "size"),
        )
        .rename(columns={"data": "locality"})
    )
    wave_fields = labeled_waves[
        [
            "wave_id",
            "wave_start",
            "wave_end",
            "zone_count",
            "wave_family_proxy",
            "reported_source",
            "source_label_confidence",
        ]
    ]
    result = locality_rows.merge(wave_fields, on="wave_id", how="left")
    return result[columns].sort_values(
        ["wave_id", "first_alert_time", "locality"]
    ).reset_index(drop=True)


def _naive_local(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed.replace(tzinfo=None)


def _time_window(time_basis: str) -> tuple[float, float]:
    """Allowed (wave start - event time) range, in minutes."""
    return {
        "explicit_siren_time": (-5.0, 5.0),
        "detection_before_alert": (-3.0, 18.0),
        "confirmation_after_alert": (-60.0, 5.0),
        "launch_report": (-45.0, 12.0),
    }.get(time_basis, (-30.0, 10.0))


def _time_score(delta_minutes: float, time_basis: str) -> float:
    if time_basis == "explicit_siren_time":
        return max(0.0, 1.0 - abs(delta_minutes) / 8.0)
    if time_basis == "detection_before_alert":
        if delta_minutes >= 0:
            return max(0.0, 1.0 - delta_minutes / 24.0)
        return max(0.0, 0.72 - abs(delta_minutes) / 10.0)
    if time_basis == "confirmation_after_alert":
        if delta_minutes <= 0:
            return max(0.0, 1.0 - abs(delta_minutes) / 75.0)
        return max(0.0, 0.55 - delta_minutes / 12.0)
    return max(0.0, 1.0 - abs(delta_minutes) / 55.0)


def match_events_to_waves(events: pd.DataFrame, waves: pd.DataFrame) -> pd.DataFrame:
    """Match from timestamps only; footprint/family is deliberately not used."""
    records = []
    wave_times = waves[["wave_id", "wave_start", "zone_count", "wave_family_proxy"]]

    for event in events.itertuples(index=False):
        base = {
            "event_id": event.event_id,
            "post_id": event.post_id,
            "reported_source": event.reported_source,
            "threat_type": event.threat_type,
            "time_basis": event.time_basis,
            "event_time_local": event.event_time_local,
            "source_url": event.source_url,
            "text": event.text,
        }
        if not bool(event.eligible_for_missile_wave_match):
            records.append(
                {
                    **base,
                    "matched_wave_id": np.nan,
                    "wave_start": None,
                    "zone_count": np.nan,
                    "wave_family_proxy": None,
                    "time_delta_minutes": np.nan,
                    "match_score": np.nan,
                    "next_best_score": np.nan,
                    "match_status": "not_eligible",
                }
            )
            continue

        event_time = _naive_local(event.event_time_local)
        lower, upper = _time_window(event.time_basis)
        deltas = (wave_times["wave_start"] - event_time).dt.total_seconds() / 60
        candidates = wave_times.loc[(deltas >= lower) & (deltas <= upper)].copy()
        candidates["time_delta_minutes"] = deltas.loc[candidates.index]
        candidates["match_score"] = candidates["time_delta_minutes"].map(
            lambda value: _time_score(float(value), event.time_basis)
        )
        candidates = candidates.sort_values(
            ["match_score", "time_delta_minutes"], ascending=[False, True]
        )

        if candidates.empty:
            records.append(
                {
                    **base,
                    "matched_wave_id": np.nan,
                    "wave_start": None,
                    "zone_count": np.nan,
                    "wave_family_proxy": None,
                    "time_delta_minutes": np.nan,
                    "match_score": np.nan,
                    "next_best_score": np.nan,
                    "match_status": "unmatched",
                }
            )
            continue

        best = candidates.iloc[0]
        next_score = float(candidates.iloc[1]["match_score"]) if len(candidates) > 1 else 0.0
        score = float(best["match_score"])
        margin = score - next_score
        if event.time_basis == "explicit_siren_time" and abs(best["time_delta_minutes"]) <= 3:
            status = "high"
        elif score >= 0.82 and margin >= 0.12:
            status = "high"
        elif score >= 0.55 and margin >= 0.06:
            status = "medium"
        else:
            status = "ambiguous"
        records.append(
            {
                **base,
                "matched_wave_id": int(best["wave_id"]),
                "wave_start": best["wave_start"],
                "zone_count": int(best["zone_count"]),
                "wave_family_proxy": best["wave_family_proxy"],
                "time_delta_minutes": float(best["time_delta_minutes"]),
                "match_score": score,
                "next_best_score": next_score,
                "match_status": status,
            }
        )
    return pd.DataFrame(records)


def apply_manual_overrides(matches: pd.DataFrame, overrides_path: Path) -> pd.DataFrame:
    if not overrides_path.exists() or overrides_path.stat().st_size == 0:
        return matches
    overrides = pd.read_csv(overrides_path)
    if overrides.empty:
        return matches
    result = matches.copy()
    for override in overrides.itertuples(index=False):
        selected = result["event_id"] == override.event_id
        if not selected.any():
            continue
        if pd.notna(getattr(override, "override_wave_id", np.nan)):
            result.loc[selected, "matched_wave_id"] = int(override.override_wave_id)
        if pd.notna(getattr(override, "override_source", np.nan)):
            result.loc[selected, "reported_source"] = override.override_source
        result.loc[selected, "match_status"] = "manual"
    return result


def aggregate_wave_labels(waves: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    accepted = matches.loc[matches["match_status"].isin(["high", "medium", "manual"])].copy()
    evidence = []
    for wave_id, group in accepted.groupby("matched_wave_id"):
        sources = sorted(set(group["reported_source"]) - {"unknown"})
        if not sources:
            label = "unknown"
        elif len(sources) == 1:
            label = sources[0]
        else:
            label = "mixed"
        if "manual" in set(group["match_status"]):
            confidence = "manual"
        elif label == "mixed":
            confidence = "conflict"
        elif "high" in set(group["match_status"]):
            confidence = "high"
        else:
            confidence = "medium"
        evidence.append(
            {
                "wave_id": int(wave_id),
                "reported_source": label,
                "source_label_confidence": confidence,
                "source_evidence_count": len(group),
                "source_event_ids": ";".join(group["event_id"].astype(str)),
                "source_post_ids": ";".join(group["post_id"].astype(str)),
                "source_urls": ";".join(dict.fromkeys(group["source_url"].astype(str))),
            }
        )
    labels = pd.DataFrame(evidence)
    result = waves.merge(labels, on="wave_id", how="left")
    result["reported_source"] = result["reported_source"].fillna("unknown")
    result["source_label_confidence"] = result["source_label_confidence"].fillna("none")
    result["source_evidence_count"] = result["source_evidence_count"].fillna(0).astype(int)
    return result


def make_validation_sample(matches: pd.DataFrame, per_status: int = 15) -> pd.DataFrame:
    samples = []
    for status in ["high", "medium", "ambiguous", "unmatched"]:
        group = matches.loc[matches["match_status"] == status]
        if group.empty:
            continue
        samples.append(
            group.sample(min(per_status, len(group)), random_state=20260721)
        )
    if not samples:
        return matches.head(0)
    sample = pd.concat(samples, ignore_index=True)
    sample["manual_correct"] = ""
    sample["manual_source"] = ""
    sample["manual_wave_id"] = ""
    sample["manual_note"] = ""
    return sample

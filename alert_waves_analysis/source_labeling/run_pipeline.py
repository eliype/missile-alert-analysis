#!/usr/bin/env python3
"""Build a confidence-aware alert-wave source dataset from official IDF posts."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ANALYSIS_ROOT = HERE.parent
PROJECT_ROOT = ANALYSIS_ROOT.parent

from event_rules import extract_events
from idf_telegram import crawl_messages
from wave_matching import (
    aggregate_wave_labels,
    apply_manual_overrides,
    make_validation_sample,
    make_wave_localities,
    match_events_to_waves,
    reconstruct_missile_waves,
)


DEFAULT_ALERTS = PROJECT_ROOT / "data" / "processed" / "second_war.csv"
RAW_DIR = ANALYSIS_ROOT / "data" / "raw"
PROCESSED_DIR = HERE / "outputs"
SHARED_PROCESSED_DIR = ANALYSIS_ROOT / "data" / "processed"
MANUAL_DIR = HERE / "manual"
VALIDATION_COLUMNS = [
    "manual_correct",
    "manual_source",
    "manual_wave_id",
    "manual_note",
]


def preserve_validation_reviews(
    validation: pd.DataFrame, existing_path: Path
) -> pd.DataFrame:
    """Carry completed review fields across deterministic pipeline reruns."""
    if not existing_path.exists():
        return validation
    existing = pd.read_csv(existing_path, dtype=str).drop_duplicates("event_id")
    if existing.empty or "event_id" not in existing:
        return validation
    indexed = existing.set_index("event_id")
    result = validation.copy()
    for column in VALIDATION_COLUMNS:
        if column not in indexed:
            continue
        saved = result["event_id"].map(indexed[column])
        result[column] = saved.where(saved.notna(), result[column])
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alerts", type=Path, default=DEFAULT_ALERTS)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2026, 2, 28))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2026, 6, 17))
    parser.add_argument("--wave-gap", type=int, default=8)
    parser.add_argument(
        "--initial-before",
        type=int,
        default=18_800,
        help="Telegram post cursor just after the requested end date",
    )
    parser.add_argument("--refresh", action="store_true", help="Redownload cached HTML pages")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Reuse alert_waves_analysis/data/raw/idf_telegram_messages.csv without network access",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    SHARED_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MANUAL_DIR.mkdir(parents=True, exist_ok=True)

    messages_path = RAW_DIR / "idf_telegram_messages.csv"
    if args.offline:
        if not messages_path.exists():
            raise FileNotFoundError(
                f"Offline mode requires an existing message snapshot: {messages_path}"
            )
        messages = pd.read_csv(messages_path)
    else:
        messages = crawl_messages(
            start_date=args.start,
            end_date=args.end,
            cache_dir=RAW_DIR / "idf_telegram_pages",
            initial_before=args.initial_before,
            refresh=args.refresh,
        )
        messages.to_csv(messages_path, index=False)

    events = extract_events(messages)
    missile_rows, waves = reconstruct_missile_waves(args.alerts, args.wave_gap)
    start_timestamp = pd.Timestamp(args.start)
    end_timestamp = pd.Timestamp(args.end) + pd.Timedelta(days=1)
    waves = waves.loc[
        (waves["wave_start"] >= start_timestamp)
        & (waves["wave_start"] < end_timestamp)
    ].copy()
    valid_wave_ids = set(waves["wave_id"])
    missile_rows = missile_rows.loc[missile_rows["wave_id"].isin(valid_wave_ids)].copy()

    matches = match_events_to_waves(events, waves)
    matches = apply_manual_overrides(matches, MANUAL_DIR / "event_overrides.csv")
    labeled_waves = aggregate_wave_labels(waves, matches)
    wave_localities = make_wave_localities(missile_rows, labeled_waves)
    validation_path = PROCESSED_DIR / "validation_sample.csv"
    validation = preserve_validation_reviews(
        make_validation_sample(matches), validation_path
    )

    events.to_csv(SHARED_PROCESSED_DIR / "idf_incoming_threats.csv", index=False)
    matches.to_csv(PROCESSED_DIR / "event_wave_matches.csv", index=False)
    labeled_waves.to_csv(SHARED_PROCESSED_DIR / "labeled_alert_waves.csv", index=False)
    wave_localities.to_csv(SHARED_PROCESSED_DIR / "wave_localities.csv", index=False)
    validation.to_csv(validation_path, index=False)

    accepted = labeled_waves.loc[labeled_waves["reported_source"] != "unknown"]
    cross_tab = pd.crosstab(
        accepted["reported_source"],
        accepted["wave_family_proxy"],
        margins=True,
    )
    cross_tab.to_csv(PROCESSED_DIR / "source_by_wave_family.csv")

    source_metrics = []
    for source, group in accepted.groupby("reported_source"):
        source_metrics.append(
            {
                "reported_source": source,
                "labeled_waves": len(group),
                "broad_wave_fraction": (group["wave_family_proxy"] == "broad").mean(),
                "median_zone_count": group["zone_count"].median(),
                "mean_zone_count": group["zone_count"].mean(),
            }
        )
    pd.DataFrame(source_metrics).to_csv(
        PROCESSED_DIR / "source_wave_summary.csv", index=False
    )

    # Test the broad/localized proxy only after matching. It was not used by
    # the matcher, so this is an out-of-feature association test rather than a
    # circular label definition.
    two_source = accepted.loc[accepted["reported_source"].isin(["iran", "lebanon"])]
    table = pd.crosstab(two_source["reported_source"], two_source["wave_family_proxy"])
    for row in ["iran", "lebanon"]:
        for column in ["broad", "localized"]:
            if row not in table.index:
                table.loc[row, column] = 0
            elif column not in table.columns:
                table.loc[row, column] = 0
    table = table.reindex(index=["iran", "lebanon"], columns=["broad", "localized"])
    odds_ratio, fisher_p = stats.fisher_exact(table.to_numpy())
    iran_broad = int(table.loc["iran", "broad"])
    iran_localized = int(table.loc["iran", "localized"])
    lebanon_broad = int(table.loc["lebanon", "broad"])
    lebanon_localized = int(table.loc["lebanon", "localized"])
    association = pd.DataFrame(
        [
            {
                "iran_broad": iran_broad,
                "iran_localized": iran_localized,
                "lebanon_broad": lebanon_broad,
                "lebanon_localized": lebanon_localized,
                "p_broad_given_iran": iran_broad / max(iran_broad + iran_localized, 1),
                "p_localized_given_lebanon": lebanon_localized / max(lebanon_broad + lebanon_localized, 1),
                "p_iran_given_broad": iran_broad / max(iran_broad + lebanon_broad, 1),
                "p_lebanon_given_localized": lebanon_localized / max(iran_localized + lebanon_localized, 1),
                "broad_iran_vs_lebanon_odds_ratio": odds_ratio,
                "fisher_exact_p_value": fisher_p,
            }
        ]
    )
    association.to_csv(PROCESSED_DIR / "source_family_association.csv", index=False)

    # The primary phase ends after 2026-04-08, the final day in the longest
    # continuous run of eligible official Iran incoming-threat reports. The
    # main analysis computes this boundary directly and audits confidence
    # strata separately.
    primary_phase_end_exclusive = pd.Timestamp("2026-04-09")
    iran_active = labeled_waves.loc[
        (labeled_waves["reported_source"] == "iran")
        & (labeled_waves["wave_start"] < primary_phase_end_exclusive)
    ].copy()
    if not iran_active.empty:
        iran_start = iran_active["wave_start"].dt.floor("D").min()
        iran_active["day_index"] = (
            iran_active["wave_start"].dt.floor("D") - iran_start
        ).dt.days
        iran_daily = iran_active.groupby("day_index", as_index=False).agg(
            date=("wave_start", lambda x: x.min().date().isoformat()),
            labeled_iran_waves=("wave_id", "size"),
            mean_zone_count=("zone_count", "mean"),
            median_zone_count=("zone_count", "median"),
            broad_fraction=("wave_family_proxy", lambda x: (x == "broad").mean()),
        )
        iran_daily.to_csv(PROCESSED_DIR / "iran_labeled_daily_evolution.csv", index=False)
        iran_rho, iran_p = stats.spearmanr(
            iran_daily["day_index"], iran_daily["mean_zone_count"]
        )
    else:
        iran_rho, iran_p = float("nan"), float("nan")

    summary = {
        "parameters": {
            "alerts": str(args.alerts),
            "start": args.start.isoformat(),
            "end": args.end.isoformat(),
            "wave_gap_minutes": args.wave_gap,
            "initial_telegram_before": args.initial_before,
            "matching_features": ["official event time", "time-basis-specific window"],
            "features_not_used_for_matching": [
                "zone_count",
                "wave_family_proxy",
                "pre-warning coverage",
                "geography",
            ],
        },
        "counts": {
            "official_messages": int(len(messages)),
            "candidate_incoming_threats": int(len(events)),
            "missile_alert_rows": int(len(missile_rows)),
            "reconstructed_waves": int(len(waves)),
            "wave_locality_rows": int(len(wave_localities)),
            "unique_localities": int(wave_localities["locality"].nunique()),
            "source_labeled_waves": int(len(accepted)),
            "source_label_coverage": float(len(accepted) / len(waves)) if len(waves) else 0,
            "manual_validation_reviewed_events": int(
                validation["manual_correct"].fillna("").astype(str).str.strip().ne("").sum()
            ),
        },
        "event_match_status": {
            str(key): int(value)
            for key, value in matches["match_status"].value_counts().to_dict().items()
        },
        "wave_source_labels": {
            str(key): int(value)
            for key, value in labeled_waves["reported_source"].value_counts().to_dict().items()
        },
        "source_family_association": association.iloc[0].to_dict(),
        "iran_labeled_primary_phase_evolution": {
            "end_date_inclusive": "2026-04-08",
            "day_vs_daily_mean_zone_count_spearman_rho": float(iran_rho),
            "spearman_p_value": float(iran_p),
        },
    }
    (PROCESSED_DIR / "pipeline_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, indent=2))
    print("\nSource by wave family:")
    print(cross_tab.to_string())


if __name__ == "__main__":
    main()

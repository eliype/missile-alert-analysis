#!/usr/bin/env python3
"""Analyze how missile-alert wave footprint and short-term targeting evolve.

The script is intentionally self-contained and reproducible. It:
1. keeps only missile alerts (category 1),
2. groups alerts into temporal waves,
3. discovers localized and broad wave families from footprint size,
4. defines the primary phase from continuous official Iran-threat reporting,
5. measures footprint change and target persistence with sensitivity checks,
6. writes auditable tables and lightweight SVG/PNG figures.

Run from the repository root:
    .venv/bin/python alert_waves_analysis/analysis.py
"""

from __future__ import annotations

import csv
import html
import json
import math
import os
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
FIGURES = HERE / "outputs" / "figures"
TABLES = HERE / "outputs" / "tables"
SOURCE_EVENTS = HERE / "data" / "processed" / "idf_incoming_threats.csv"
LABELED_WAVES = HERE / "data" / "processed" / "labeled_alert_waves.csv"

WAR_FILES = {
    "first_war": ROOT / "data" / "processed" / "first_war.csv",
    "second_war": ROOT / "data" / "processed" / "second_war.csv",
}

DEFAULT_GAP_MINUTES = 8
GAP_SENSITIVITY = [3, 5, 8, 10, 15]
PHASE_END_SENSITIVITY_DAYS = [-7, -3, 0, 3, 7]
PHASE_START_TRIM_DAYS = [0, 1, 3, 7]
CLUSTER_K_VALUES = list(range(1, 13))
CLUSTER_STABILITY_SEEDS = list(range(20))
MIN_CLUSTER_FRACTION = 0.05
RANDOM_SEED = 20260721
N_PERMUTATIONS = 2_000
N_SENSITIVITY_PERMUTATIONS = 500

COLORS = {
    "ink": "#172033",
    "muted": "#667085",
    "grid": "#d9e0ea",
    "localized": "#2f80ed",
    "broad": "#e05a47",
    "null": "#aeb8c5",
    "accent": "#7b61a8",
    "background": "#ffffff",
}


def load_war(path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    """Load one war CSV, retain missile alerts, and remove exact repeats."""
    raw = pd.read_csv(path)
    missile = raw.loc[raw["category"] == 1].copy()
    missile["timestamp"] = pd.to_datetime(missile["alertDate"], errors="raise")
    duplicate_count = int(missile.duplicated(["alertDate", "data"]).sum())
    missile = missile.drop_duplicates(["alertDate", "data"]).sort_values("timestamp")
    diagnostics = {
        "raw_rows": int(len(raw)),
        "missile_rows_before_dedup": int((raw["category"] == 1).sum()),
        "missile_rows_after_dedup": int(len(missile)),
        "duplicate_missile_zone_timestamps_removed": duplicate_count,
    }
    return missile.reset_index(drop=True), diagnostics


def build_waves(
    missile: pd.DataFrame, gap_minutes: int
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, set[str]]]:
    """Join consecutive alerts when their gap is at most ``gap_minutes``."""
    rows = missile.sort_values("timestamp").copy()
    rows["new_wave"] = (
        rows["timestamp"].diff().dt.total_seconds().gt(gap_minutes * 60).fillna(True)
    )
    rows["wave_id"] = rows["new_wave"].cumsum().astype(int)

    waves = (
        rows.groupby("wave_id", as_index=False)
        .agg(
            start=("timestamp", "min"),
            end=("timestamp", "max"),
            alert_rows=("data", "size"),
            zone_count=("data", "nunique"),
            burst_count=("timestamp", "nunique"),
        )
        .sort_values("start")
    )
    waves["duration_minutes"] = (
        waves["end"] - waves["start"]
    ).dt.total_seconds() / 60
    waves["gap_from_previous_hours"] = waves["start"].diff().dt.total_seconds() / 3600
    start_day = waves["start"].dt.floor("D").min()
    waves["day_index"] = (waves["start"].dt.floor("D") - start_day).dt.days
    waves["date"] = waves["start"].dt.date.astype(str)
    waves["hour"] = waves["start"].dt.hour + waves["start"].dt.minute / 60

    zone_sets = {
        int(wave_id): set(group["data"])
        for wave_id, group in rows.groupby("wave_id", sort=True)
    }
    return rows, waves, zone_sets


def discover_family_cutoff(waves: pd.DataFrame) -> tuple[int, dict[str, float]]:
    """Cluster log footprint into two interpretable families using 1-D k-means."""
    x = np.log1p(waves["zone_count"].to_numpy()).reshape(-1, 1)
    model = KMeans(n_clusters=2, random_state=RANDOM_SEED, n_init=50).fit(x)
    centers = model.cluster_centers_.ravel()
    localized_label = int(np.argmin(centers))
    localized_sizes = waves.loc[model.labels_ == localized_label, "zone_count"]
    broad_sizes = waves.loc[model.labels_ != localized_label, "zone_count"]
    cutoff = int(localized_sizes.max()) + 1
    details = {
        "localized_geometric_center_zones": float(np.expm1(centers[localized_label])),
        "broad_geometric_center_zones": float(
            np.expm1(centers[1 - localized_label])
        ),
        "largest_localized_cluster_wave": int(localized_sizes.max()),
        "smallest_broad_cluster_wave": int(broad_sizes.min()),
    }
    return cutoff, details


def compute_cluster_validation(
    waves: pd.DataFrame,
    k_values: list[int] | tuple[int, ...] = tuple(CLUSTER_K_VALUES),
    stability_seeds: list[int] | tuple[int, ...] = tuple(CLUSTER_STABILITY_SEEDS),
    minimum_cluster_fraction: float = MIN_CLUSTER_FRACTION,
) -> pd.DataFrame:
    """Perform relative validation and initialization-stability checks."""
    x = np.log1p(waves["zone_count"].to_numpy()).reshape(-1, 1)
    records = []
    previous_sse: float | None = None
    baseline_sse: float | None = None
    for k in k_values:
        if k < 1 or k >= len(x):
            raise ValueError("Each k must be at least 1 and smaller than sample size")
        model = KMeans(
            n_clusters=k,
            random_state=RANDOM_SEED,
            n_init=50,
            init="k-means++",
        ).fit(x)
        if baseline_sse is None:
            baseline_sse = float(model.inertia_)
        labels = model.labels_
        cluster_sizes = sorted(np.bincount(labels).tolist(), reverse=True)
        silhouette = math.nan
        mean_ari = math.nan
        if k > 1:
            silhouette = float(silhouette_score(x, labels, metric="euclidean"))
            repeated_labels = [
                KMeans(
                    n_clusters=k,
                    random_state=seed,
                    n_init=1,
                    init="k-means++",
                ).fit_predict(x)
                for seed in stability_seeds
            ]
            agreements = [
                adjusted_rand_score(repeated_labels[first], repeated_labels[second])
                for first in range(len(repeated_labels))
                for second in range(first)
            ]
            mean_ari = float(np.mean(agreements)) if agreements else math.nan
        inertia = float(model.inertia_)
        minimum_fraction = min(cluster_sizes) / len(x)
        records.append(
            {
                "k": k,
                "within_cluster_sse": inertia,
                "sse_relative_to_k1": inertia / baseline_sse,
                "sse_reduction_from_previous_k": (
                    math.nan if previous_sse is None else 1 - inertia / previous_sse
                ),
                "silhouette_score": silhouette,
                "mean_pairwise_adjusted_rand": mean_ari,
                "minimum_cluster_size": min(cluster_sizes),
                "minimum_cluster_fraction": minimum_fraction,
                "eligible_for_family_selection": (
                    minimum_fraction >= minimum_cluster_fraction
                ),
                "cluster_sizes_descending": ";".join(map(str, cluster_sizes)),
            }
        )
        previous_sse = inertia
    return pd.DataFrame(records)


def add_family(waves: pd.DataFrame, cutoff: int) -> pd.DataFrame:
    result = waves.copy()
    result["family"] = np.where(
        result["zone_count"] >= cutoff, "broad", "localized"
    )
    return result


def find_three_regimes(daily_counts: pd.Series, min_days: int = 5) -> tuple[int, int]:
    """Find two least-squares change points in log(1 + daily alerts)."""
    y = np.log1p(daily_counts.to_numpy(dtype=float))

    def cost(start: int, end: int) -> float:
        segment = y[start:end]
        return float(((segment - segment.mean()) ** 2).sum())

    best_cost = math.inf
    best = (min_days, len(y) - min_days)
    for first in range(min_days, len(y) - 2 * min_days + 1):
        for second in range(first + min_days, len(y) - min_days + 1):
            candidate = (
                cost(0, first) + cost(first, second) + cost(second, len(y))
            )
            if candidate < best_cost:
                best_cost = candidate
                best = (first, second)
    return best


def longest_continuous_source_phase(
    events: pd.DataFrame, source: str = "iran"
) -> tuple[pd.Timestamp, pd.Timestamp, int]:
    """Return the longest run of calendar days with eligible source reports.

    The returned end timestamp is exclusive. Source reports are used only to
    define the analysis window; alert footprint and wave family are not used.
    """
    required = {
        "reported_source",
        "eligible_for_missile_wave_match",
        "event_time_local",
    }
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"Source event data is missing columns: {sorted(missing)}")

    eligible = events["eligible_for_missile_wave_match"]
    if eligible.dtype != bool:
        eligible = eligible.astype(str).str.lower().eq("true")
    selected = events.loc[
        events["reported_source"].eq(source) & eligible,
        "event_time_local",
    ].dropna()
    if selected.empty:
        raise ValueError(f"No eligible official events found for source={source!r}")

    local_days = (
        pd.to_datetime(selected, utc=True, errors="raise")
        .dt.tz_convert("Asia/Jerusalem")
        .dt.tz_localize(None)
        .dt.floor("D")
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    runs: list[list[pd.Timestamp]] = []
    current = [local_days[0]]
    for day in local_days[1:]:
        if day - current[-1] == pd.Timedelta(days=1):
            current.append(day)
        else:
            runs.append(current)
            current = [day]
    runs.append(current)
    longest = max(runs, key=lambda run: (len(run), -run[0].value))
    return longest[0], longest[-1] + pd.Timedelta(days=1), len(longest)


def compute_phase_boundary_sensitivity(
    waves: pd.DataFrame,
    calendar_start: pd.Timestamp,
    primary_end_day: int,
) -> pd.DataFrame:
    """Repeat the main trend around nearby phase-end definitions."""
    records = []
    for shift in PHASE_END_SENSITIVITY_DAYS:
        end_day = primary_end_day + shift
        if end_day < 14:
            continue
        active = waves.loc[waves["day_index"] < end_day]
        daily = active.groupby("day_index").agg(
            mean_footprint=("zone_count", "mean"),
            waves=("wave_id", "size"),
        ).reindex(range(end_day))
        valid = daily["mean_footprint"].notna()
        rho, p_value = stats.spearmanr(
            daily.index[valid], daily.loc[valid, "mean_footprint"]
        )
        early = daily.iloc[:7]
        late = daily.iloc[-7:]
        early_footprint = float(early["mean_footprint"].mean())
        late_footprint = float(late["mean_footprint"].mean())
        records.append(
            {
                "end_shift_days": shift,
                "phase_days": end_day,
                "phase_end_inclusive": (
                    calendar_start + pd.Timedelta(days=end_day - 1)
                ).date().isoformat(),
                "days_with_waves": int(valid.sum()),
                "day_vs_daily_mean_footprint_spearman_rho": float(rho),
                "spearman_p_value": float(p_value),
                "first_7_days_mean_footprint": early_footprint,
                "last_7_days_mean_footprint": late_footprint,
                "early_to_late_footprint_reduction": (
                    (early_footprint - late_footprint) / early_footprint
                ),
                "first_7_days_mean_waves_per_day": float(early["waves"].mean()),
                "last_7_days_mean_waves_per_day": float(late["waves"].mean()),
            }
        )
    return pd.DataFrame(records)


def compute_phase_start_sensitivity(
    daily: pd.DataFrame,
    trim_days: list[int] | tuple[int, ...] = tuple(PHASE_START_TRIM_DAYS),
) -> pd.DataFrame:
    """Check whether unusually broad opening days alone create the trend."""
    records = []
    for trim in trim_days:
        selected = daily.iloc[trim:].dropna(subset=["mean_zones_per_wave"])
        if len(selected) < 3:
            raise ValueError("At least three daily observations are required")
        rho, p_value = stats.spearmanr(
            selected["day_index"], selected["mean_zones_per_wave"]
        )
        records.append(
            {
                "opening_days_excluded": trim,
                "analysis_start_date": str(selected.iloc[0]["date"]),
                "remaining_days": int(len(selected)),
                "day_vs_daily_mean_footprint_spearman_rho": float(rho),
                "spearman_p_value": float(p_value),
            }
        )
    return pd.DataFrame(records)


def make_wave_reality_checks(
    waves: pd.DataFrame,
    zone_sets: dict[int, set[str]],
    family_cutoff: int,
) -> pd.DataFrame:
    """Build a small deterministic packet for inspecting reconstruction edges."""
    selections = {
        "largest_footprint": waves.sort_values(
            ["zone_count", "start"], ascending=[False, True]
        ).head(8),
        "longest_duration": waves.sort_values(
            ["duration_minutes", "start"], ascending=[False, True]
        ).head(5),
        "just_below_family_cutoff": waves.loc[
            waves["zone_count"] < family_cutoff
        ].sort_values(["zone_count", "start"], ascending=[False, True]).head(5),
        "just_above_family_cutoff": waves.loc[
            waves["zone_count"] >= family_cutoff
        ].sort_values(["zone_count", "start"], ascending=[True, True]).head(5),
    }
    records = []
    for check_type, selected in selections.items():
        for row in selected.itertuples(index=False):
            localities = sorted(zone_sets[int(row.wave_id)])
            records.append(
                {
                    "check_type": check_type,
                    "wave_id": int(row.wave_id),
                    "start": row.start,
                    "end": row.end,
                    "duration_minutes": float(row.duration_minutes),
                    "zone_count": int(row.zone_count),
                    "burst_count": int(row.burst_count),
                    "family": row.family,
                    "sample_localities": "; ".join(localities[:12]),
                    "sample_localities_truncated": len(localities) > 12,
                }
            )
    return pd.DataFrame(records)


def compute_source_trend_diagnostics(
    labeled_waves: pd.DataFrame,
    phase_start: pd.Timestamp,
    phase_end_exclusive: pd.Timestamp,
) -> pd.DataFrame:
    """Show how the Iran-labeled trend changes with match confidence."""
    waves = labeled_waves.copy()
    waves["wave_start"] = pd.to_datetime(waves["wave_start"], errors="raise")
    waves = waves.loc[
        waves["reported_source"].eq("iran")
        & waves["wave_start"].ge(phase_start)
        & waves["wave_start"].lt(phase_end_exclusive)
    ].copy()
    waves["day_index"] = (
        waves["wave_start"].dt.floor("D") - phase_start
    ).dt.days

    records = []
    confidence_groups: list[tuple[str, pd.DataFrame]] = [
        ("all_accepted", waves),
        ("high", waves.loc[waves["source_label_confidence"].eq("high")]),
        ("medium", waves.loc[waves["source_label_confidence"].eq("medium")]),
    ]
    for label, selected in confidence_groups:
        daily = selected.groupby("day_index").agg(
            mean_footprint=("zone_count", "mean"),
            median_footprint=("zone_count", "median"),
        )
        mean_rho, mean_p = stats.spearmanr(daily.index, daily["mean_footprint"])
        median_rho, median_p = stats.spearmanr(
            daily.index, daily["median_footprint"]
        )
        records.append(
            {
                "source_label_confidence": label,
                "waves": int(len(selected)),
                "days_with_labels": int(len(daily)),
                "broad_fraction": float(selected["wave_family_proxy"].eq("broad").mean()),
                "day_vs_daily_mean_footprint_spearman_rho": float(mean_rho),
                "mean_footprint_spearman_p": float(mean_p),
                "day_vs_daily_median_footprint_spearman_rho": float(median_rho),
                "median_footprint_spearman_p": float(median_p),
            }
        )
    return pd.DataFrame(records)


def add_warning_features(
    raw_path: Path, missile_rows: pd.DataFrame, waves: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    """Match each alerted zone to its most recent preliminary warning within 60 min."""
    all_rows = pd.read_csv(raw_path)
    all_rows["timestamp"] = pd.to_datetime(all_rows["alertDate"], errors="raise")
    description = all_rows["category_desc"].fillna("")
    is_pre_warning = (all_rows["category"] == 14) & description.str.contains(
        "בדקות הקרובות|התקרבו למרחב מוגן", regex=True
    )
    warnings = all_rows.loc[is_pre_warning, ["data", "timestamp"]].copy()
    warning_lookup = {
        zone: np.sort(group["timestamp"].to_numpy(dtype="datetime64[ns]"))
        for zone, group in warnings.groupby("data")
    }

    leads: list[float] = []
    for row in missile_rows.itertuples(index=False):
        times = warning_lookup.get(row.data)
        lead = math.nan
        if times is not None:
            alert_time = np.datetime64(row.timestamp)
            position = int(np.searchsorted(times, alert_time, side="right") - 1)
            if position >= 0:
                minutes = float((alert_time - times[position]) / np.timedelta64(1, "m"))
                if 0 <= minutes <= 60:
                    lead = minutes
        leads.append(lead)

    matched = missile_rows[["wave_id", "data", "timestamp"]].copy()
    matched["warning_lead_minutes"] = leads
    # A zone can receive multiple alerts inside a wave; use its first one.
    matched = matched.sort_values("timestamp").drop_duplicates(["wave_id", "data"])
    by_wave = matched.groupby("wave_id").agg(
        warning_coverage=("warning_lead_minutes", lambda x: x.notna().mean()),
        median_warning_lead_minutes=("warning_lead_minutes", "median"),
    )
    result = waves.merge(by_wave, on="wave_id", how="left")
    first_warning = None if warnings.empty else warnings["timestamp"].min()
    return result, first_warning


def daily_metrics(waves: pd.DataFrame, total_days: int | None = None) -> pd.DataFrame:
    grouped = waves.groupby("day_index").agg(
        date=("date", "first"),
        waves=("wave_id", "size"),
        mean_zones_per_wave=("zone_count", "mean"),
        median_zones_per_wave=("zone_count", "median"),
        mean_duration_minutes=("duration_minutes", "mean"),
        broad_fraction=("family", lambda x: (x == "broad").mean()),
        broad_wave_count=("family", lambda x: (x == "broad").sum()),
    )
    broad_median = (
        waves.loc[waves["family"] == "broad"]
        .groupby("day_index")["zone_count"]
        .median()
        .rename("median_broad_wave_zones")
    )
    grouped = grouped.join(broad_median)
    if total_days is not None:
        grouped = grouped.reindex(range(total_days))
    grouped.index.name = "day_index"
    return grouped.reset_index()


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b)


def persistence_permutation_test(
    waves: pd.DataFrame,
    zone_sets: dict[int, set[str]],
    high_intensity_end: int,
    n_permutations: int = N_PERMUTATIONS,
) -> pd.DataFrame:
    """Compare truly adjacent waves with within-day order shuffles.

    Family-specific results include a pair only when both waves are adjacent in
    the complete sequence and belong to that family. The null shuffles complete
    wave records, preserving each wave's footprint and family label.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    active = waves.loc[waves["day_index"] < high_intensity_end].copy()
    day_sequences = [
        day.sort_values("start")[["wave_id", "family"]].to_records(index=False)
        for _, day in active.groupby("day_index")
        if len(day) > 1
    ]

    def score(
        sequences: list[np.recarray],
    ) -> tuple[dict[str, float], dict[str, int]]:
        values: dict[str, list[float]] = {
            "all": [],
            "broad": [],
            "localized": [],
        }
        for sequence in sequences:
            for first, second in zip(sequence[:-1], sequence[1:]):
                overlap = jaccard(
                    zone_sets[int(first.wave_id)], zone_sets[int(second.wave_id)]
                )
                values["all"].append(overlap)
                if first.family == second.family:
                    values[str(first.family)].append(overlap)
        means = {
            family: (
                float(np.mean(family_values)) if family_values else math.nan
            )
            for family, family_values in values.items()
        }
        counts = {family: len(family_values) for family, family_values in values.items()}
        return means, counts

    observed, pair_counts = score(day_sequences)
    null_values = {family: [] for family in observed}
    for _ in range(n_permutations):
        shuffled_scores, _ = score(
            [rng.permutation(sequence) for sequence in day_sequences]
        )
        for family, value in shuffled_scores.items():
            null_values[family].append(value)

    records = []
    for family in ["all", "broad", "localized"]:
        null = np.asarray(null_values[family], dtype=float)
        null = null[~np.isnan(null)]
        if null.size == 0 or math.isnan(observed[family]):
            raise ValueError(f"Not enough adjacent {family} pairs for persistence test")
        null_mean = float(null.mean())
        null_q025, null_q975 = np.quantile(null, [0.025, 0.975])
        p_value = (1 + int((null >= observed[family]).sum())) / (
            len(null) + 1
        )
        records.append(
            {
                "family": family,
                "adjacent_pairs": pair_counts[family],
                "observed_mean_jaccard": observed[family],
                "shuffled_mean_jaccard": null_mean,
                "shuffled_mean_jaccard_q025": float(null_q025),
                "shuffled_mean_jaccard_q975": float(null_q975),
                "absolute_jaccard_lift": observed[family] - null_mean,
                "observed_to_null_ratio": observed[family] / null_mean,
                "permutation_p_value": p_value,
            }
        )
    return pd.DataFrame(records)


def compute_sensitivity(missile: pd.DataFrame, high_intensity_end: int) -> pd.DataFrame:
    records = []
    for gap in GAP_SENSITIVITY:
        _, waves, _ = build_waves(missile, gap)
        active = waves.loc[waves["day_index"] < high_intensity_end]
        daily = active.groupby("day_index")["zone_count"].mean().reindex(
            range(high_intensity_end)
        )
        valid = daily.notna()
        rho, p_value = stats.spearmanr(daily.index[valid], daily.loc[valid])
        early = float(daily.iloc[:7].mean())
        late = float(daily.iloc[-7:].mean())
        records.append(
            {
                "gap_minutes": gap,
                "total_waves": len(waves),
                "high_intensity_waves": len(active),
                "day_vs_daily_mean_footprint_spearman_rho": rho,
                "spearman_p_value": p_value,
                "first_7_days_daily_mean_footprint": early,
                "last_7_days_daily_mean_footprint": late,
                "early_to_late_footprint_reduction": (early - late) / early,
            }
        )
    return pd.DataFrame(records)


def compute_persistence_gap_sensitivity(
    missile: pd.DataFrame, high_intensity_end: int
) -> pd.DataFrame:
    """Repeat the adjacency test under alternative wave-gap definitions."""
    records = []
    for gap in GAP_SENSITIVITY:
        _, waves, zone_sets = build_waves(missile, gap)
        cutoff, _ = discover_family_cutoff(waves)
        waves = add_family(waves, cutoff)
        result = persistence_permutation_test(
            waves,
            zone_sets,
            high_intensity_end,
            n_permutations=N_SENSITIVITY_PERMUTATIONS,
        )
        result.insert(0, "family_cutoff_zones", cutoff)
        result.insert(0, "gap_minutes", gap)
        records.append(result)
    return pd.concat(records, ignore_index=True)


def write_svg(name: str, width: int, height: int, body: list[str]) -> None:
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="100%" height="100%" fill="{COLORS["background"]}"/>',
        *body,
        "</svg>",
    ]
    path = FIGURES / f"{name}.svg"
    path.write_text("\n".join(svg), encoding="utf-8")
    converter = shutil.which("rsvg-convert")
    if converter:
        png_path = FIGURES / f"{name}.png"
        subprocess.run(
            [converter, str(path), "-o", str(png_path)],
            check=True,
        )


def text_element(
    x: float,
    y: float,
    value: object,
    size: int = 14,
    color: str | None = None,
    anchor: str = "start",
    weight: int = 400,
) -> str:
    safe = html.escape(str(value))
    fill = color or COLORS["ink"]
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}" '
        f'text-anchor="{anchor}">{safe}</text>'
    )


def rotated_text_element(
    x: float,
    y: float,
    value: object,
    size: int = 14,
    color: str | None = None,
    weight: int = 700,
) -> str:
    safe = html.escape(str(value))
    fill = color or COLORS["ink"]
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" transform="rotate(-90 {x:.1f} {y:.1f})" '
        f'font-family="Arial, sans-serif" font-size="{size}" font-weight="{weight}" '
        f'fill="{fill}" text-anchor="middle">{safe}</text>'
    )


def make_distribution_figure(
    waves_by_war: dict[str, pd.DataFrame],
    cutoff: int,
    validation: pd.DataFrame,
) -> None:
    # Compact, report-friendly layout: the footprint distribution is the main
    # panel on the left, with the two validation panels stacked on the right.
    width, height = 1050, 450
    left, top = 72, 115
    plot_w, plot_h = 580, 240
    bins = np.arange(0, 12.5, 0.5)
    centers = (bins[:-1] + bins[1:]) / 2
    second_waves = waves_by_war["second_war"]
    values = np.log2(second_waves["zone_count"].to_numpy())
    counts, _ = np.histogram(values, bins=bins)
    shares = counts / counts.sum()
    ymax = max(shares) * 1.15

    body = [
        text_element(
            36,
            31,
            "Two clusters capture the dominant alert-footprint split",
            25,
            weight=700,
        ),
        text_element(
            36,
            55,
            "K-means++ on log(1 + footprint)",
            14,
            COLORS["muted"],
        ),
        text_element(36, 86, "A. Footprint distribution", 17, weight=700),
    ]
    for tick in np.linspace(0, ymax, 5):
        y = top + plot_h * (1 - tick / ymax)
        body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="{COLORS["grid"]}"/>')
        body.append(text_element(left - 10, y + 5, f"{tick:.0%}", 13, COLORS["muted"], "end"))

    bar_w = plot_w / len(centers) * 0.68
    for i, value in enumerate(shares):
        x = left + (i + 0.5) * plot_w / len(centers)
        y = top + plot_h * (1 - value / ymax)
        color = COLORS["localized"] if 2 ** centers[i] < cutoff else COLORS["broad"]
        body.append(
            f'<rect x="{x-bar_w/2:.1f}" y="{y:.1f}" width="{bar_w:.1f}" '
            f'height="{top+plot_h-y:.1f}" fill="{color}" opacity="0.82"/>'
        )

    for zones in [1, 4, 16, 64, 256, 1024]:
        log_value = math.log2(zones)
        x = left + (log_value - bins[0]) / (bins[-1] - bins[0]) * plot_w
        body.append(text_element(x, top + plot_h + 24, zones, 13, COLORS["muted"], "middle"))
    cutoff_x = left + math.log2(cutoff) / (bins[-1] - bins[0]) * plot_w
    body.append(f'<line x1="{cutoff_x:.1f}" y1="{top}" x2="{cutoff_x:.1f}" y2="{top+plot_h}" stroke="{COLORS["broad"]}" stroke-width="2" stroke-dasharray="7 5"/>')
    body.append(text_element(cutoff_x + 7, top + 18, f"Split: {cutoff} localities", 13, COLORS["broad"], weight=700))
    localized_count = int(second_waves["zone_count"].lt(cutoff).sum())
    broad_count = int(second_waves["zone_count"].ge(cutoff).sum())
    body.append(
        f'<circle cx="{left+198}" cy="{top-31}" r="5" fill="{COLORS["localized"]}"/>'
    )
    body.append(
        text_element(
            left + 209,
            top - 26,
            f"Localized: 1-{cutoff-1} localities ({localized_count:,} waves)",
            12,
            COLORS["localized"],
            weight=700,
        )
    )
    body.append(
        f'<circle cx="{left+430}" cy="{top-31}" r="5" fill="{COLORS["broad"]}"/>'
    )
    body.append(
        text_element(
            left + 441,
            top - 26,
            f"Broad: {cutoff}+ localities ({broad_count:,} waves)",
            12,
            COLORS["broad"],
            weight=700,
        )
    )
    body.append(text_element(left + plot_w / 2, 416, "Distinct alerted localities per wave (log scale)", 14, anchor="middle", weight=700))
    body.append(rotated_text_element(21, top + plot_h / 2, "Share of waves", 14))
    body.append(
        f'<line x1="690" y1="76" x2="690" y2="430" stroke="{COLORS["grid"]}"/>'
    )

    k_values = validation["k"].astype(int).tolist()
    eligible_by_k = validation.set_index("k")[
        "eligible_for_family_selection"
    ].to_dict()
    sse_values = (validation["sse_relative_to_k1"] * 100).tolist()
    sse_x, sse_y, sse_w, sse_h = 748, 102, 270, 98
    body.append(text_element(716, 86, "B. SSE elbow", 17, weight=700))
    for tick in [0, 50, 100]:
        y = sse_y + sse_h * (1 - tick / 100)
        body.append(
            f'<line x1="{sse_x}" y1="{y:.1f}" x2="{sse_x+sse_w}" y2="{y:.1f}" '
            f'stroke="{COLORS["grid"]}"/>'
        )
        body.append(
            text_element(sse_x - 8, y + 5, f"{tick}%", 12, COLORS["muted"], "end")
        )
    sse_points = []
    for index, (k, value) in enumerate(zip(k_values, sse_values)):
        x = sse_x + index / max(len(k_values) - 1, 1) * sse_w
        y = sse_y + sse_h * (1 - value / 100)
        sse_points.append(f"{x:.1f},{y:.1f}")
        if k in [1, 2, 4, 6, 8, 10, 12]:
            body.append(text_element(x, sse_y + sse_h + 19, k, 11, COLORS["muted"], "middle"))
        if k == 2:
            body.append(
                text_element(x + 8, y - 7, f"k=2: {value:.1f}% remains", 11, COLORS["broad"], weight=700)
            )
    body.append(
        f'<polyline points="{" ".join(sse_points)}" fill="none" '
        f'stroke="{COLORS["accent"]}" stroke-width="3" stroke-linejoin="round"/>'
    )
    for index, (k, value) in enumerate(zip(k_values, sse_values)):
        x = sse_x + index / max(len(k_values) - 1, 1) * sse_w
        y = sse_y + sse_h * (1 - value / 100)
        color = (
            COLORS["broad"]
            if k == 2
            else (COLORS["accent"] if eligible_by_k[k] else COLORS["null"])
        )
        radius = 5 if k == 2 else 3.5
        body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{color}"/>')
    body.append(
        rotated_text_element(704, sse_y + sse_h / 2, "SSE vs. k=1", 12)
    )
    body.append(text_element(sse_x + sse_w / 2, 238, "Number of clusters, k", 12, anchor="middle", weight=700))
    k2 = validation.set_index("k").loc[2]
    body.append(
        text_element(
            sse_x + sse_w / 2,
            257,
            f"k=2 stability: mean ARI = {k2['mean_pairwise_adjusted_rand']:.3f} across {len(CLUSTER_STABILITY_SEEDS)} runs",
            11,
            COLORS["muted"],
            "middle",
        )
    )

    silhouette = validation.dropna(subset=["silhouette_score"])
    sil_k = silhouette["k"].astype(int).tolist()
    sil_values = silhouette["silhouette_score"].tolist()
    sil_x, sil_y, sil_w, sil_h = 748, 316, 270, 86
    body.append(
        text_element(716, 300, "C. Cohesion and separation", 17, weight=700)
    )
    for tick in [0, 0.5, 1.0]:
        y = sil_y + sil_h * (1 - tick)
        body.append(
            f'<line x1="{sil_x}" y1="{y:.1f}" x2="{sil_x+sil_w}" y2="{y:.1f}" '
            f'stroke="{COLORS["grid"]}"/>'
        )
        body.append(
            text_element(sil_x - 8, y + 5, f"{tick:.1f}", 12, COLORS["muted"], "end")
        )
    sil_points = []
    for index, (k, value) in enumerate(zip(sil_k, sil_values)):
        x = sil_x + index / max(len(sil_k) - 1, 1) * sil_w
        y = sil_y + sil_h * (1 - value)
        sil_points.append(f"{x:.1f},{y:.1f}")
        if k in [2, 4, 6, 8, 10, 12]:
            body.append(text_element(x, sil_y + sil_h + 19, k, 11, COLORS["muted"], "middle"))
        if k == 2:
            body.append(
                text_element(x + 8, y - 7, f"k=2: {value:.3f} (highest)", 11, COLORS["broad"], weight=700)
            )
    body.append(
        f'<polyline points="{" ".join(sil_points)}" fill="none" '
        f'stroke="{COLORS["localized"]}" stroke-width="3" stroke-linejoin="round"/>'
    )
    for index, (k, value) in enumerate(zip(sil_k, sil_values)):
        x = sil_x + index / max(len(sil_k) - 1, 1) * sil_w
        y = sil_y + sil_h * (1 - value)
        color = (
            COLORS["broad"]
            if k == 2
            else (COLORS["localized"] if eligible_by_k[k] else COLORS["null"])
        )
        radius = 5 if k == 2 else 3.5
        body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{color}"/>')
    body.append(rotated_text_element(704, sil_y + sil_h / 2, "Silhouette", 12))
    body.append(text_element(sil_x + sil_w / 2, 444, "Number of clusters, k", 12, anchor="middle", weight=700))
    write_svg("wave_size_distribution", width, height, body)


def make_timeline_figure(waves: pd.DataFrame, high_end: int, cutoff: int) -> None:
    active = waves.loc[waves["day_index"] < high_end].copy()
    width, height = 1050, 640
    left, right, top, bottom = 90, 35, 95, 80
    plot_w, plot_h = width - left - right, height - top - bottom
    y_min, y_max = 0.0, math.log10(max(active["zone_count"].max(), 1)) + 0.12
    body = [
        text_element(45, 38, "Second war: broad waves shrink and become less common", 25, weight=700),
        text_element(45, 66, f"Primary Iran-reporting phase (days 0-{high_end-1}); wave gap = {DEFAULT_GAP_MINUTES} minutes", 14, COLORS["muted"]),
    ]
    for zones in [1, 10, 100, 1000]:
        y = top + plot_h * (1 - math.log10(zones) / y_max)
        body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="{COLORS["grid"]}"/>')
        body.append(text_element(left - 12, y + 5, zones, 12, COLORS["muted"], "end"))
    for day in range(0, high_end, 5):
        x = left + day / max(high_end - 1, 1) * plot_w
        body.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+plot_h}" stroke="{COLORS["grid"]}" opacity="0.55"/>')
        body.append(text_element(x, top + plot_h + 26, day, 12, COLORS["muted"], "middle"))

    for row in active.itertuples(index=False):
        day_fraction = row.hour / 24
        x = left + (row.day_index + day_fraction) / max(high_end - 1, 1) * plot_w
        y = top + plot_h * (1 - math.log10(max(row.zone_count, 1)) / y_max)
        color = COLORS["broad"] if row.family == "broad" else COLORS["localized"]
        radius = 3.0 if row.family == "broad" else 2.2
        body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{color}" opacity="0.68"/>')

    cutoff_y = top + plot_h * (1 - math.log10(cutoff) / y_max)
    body.append(f'<line x1="{left}" y1="{cutoff_y:.1f}" x2="{left+plot_w}" y2="{cutoff_y:.1f}" stroke="{COLORS["broad"]}" stroke-dasharray="7 5"/>')
    body.append(text_element(left + 8, cutoff_y - 8, f"Broad/localized split ({cutoff} localities)", 12, COLORS["broad"], weight=700))
    body.append(text_element(left + plot_w / 2, height - 22, "Day since the first missile alert", 14, anchor="middle", weight=700))
    body.append(rotated_text_element(25, top + plot_h / 2, "Distinct alerted localities (log scale)"))
    body.append(f'<circle cx="{width-250}" cy="35" r="5" fill="{COLORS["localized"]}"/><circle cx="{width-125}" cy="35" r="5" fill="{COLORS["broad"]}"/>')
    body.append(text_element(width - 238, 40, "Localized", 13))
    body.append(text_element(width - 113, 40, "Broad", 13))
    write_svg("second_war_wave_timeline", width, height, body)


def line_panel(
    body: list[str],
    values: pd.Series,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    color: str,
    y_format,
    ymax_override: float | None = None,
) -> None:
    clean = values.astype(float)
    ymin = min(0.0, float(clean.min()))
    ymax = (
        ymax_override
        if ymax_override is not None
        else (float(clean.max()) * 1.12 if clean.max() else 1.0)
    )
    body.append(text_element(x, y - 18, title, 16, weight=700))
    for tick in np.linspace(ymin, ymax, 5):
        py = y + height * (1 - (tick - ymin) / (ymax - ymin))
        body.append(f'<line x1="{x}" y1="{py:.1f}" x2="{x+width}" y2="{py:.1f}" stroke="{COLORS["grid"]}"/>')
        body.append(text_element(x - 10, py + 4, y_format(tick), 11, COLORS["muted"], "end"))
    points = []
    n = max(len(clean) - 1, 1)
    for i, value in enumerate(clean):
        px = x + i / n * width
        py = y + height * (1 - (value - ymin) / (ymax - ymin))
        points.append(f"{px:.1f},{py:.1f}")
    body.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round"/>')


def make_evolution_figure(daily: pd.DataFrame) -> None:
    width, height = 1050, 760
    body = [
        text_element(45, 38, "The shift is both compositional and within broad waves", 25, weight=700),
        text_element(45, 66, "Daily summaries during the 40-day primary Iran-reporting phase", 14, COLORS["muted"]),
    ]
    line_panel(
        body,
        daily["median_broad_wave_zones"],
        100,
        120,
        900,
        220,
        "A. Median footprint among broad waves",
        COLORS["accent"],
        lambda value: f"{value:.0f}",
    )
    line_panel(
        body,
        daily["broad_fraction"],
        100,
        430,
        900,
        220,
        "B. Share of waves assigned to the broad family",
        COLORS["broad"],
        lambda value: f"{value:.0%}",
        1.0,
    )
    for day in range(0, len(daily), 5):
        x = 100 + day / max(len(daily) - 1, 1) * 900
        body.append(text_element(x, 682, day, 12, COLORS["muted"], "middle"))
    body.append(text_element(550, 725, "Day since the first missile alert", 14, anchor="middle", weight=700))
    write_svg("second_war_evolution", width, height, body)


def make_primary_phase_summary_figure(daily: pd.DataFrame) -> None:
    """Show the main footprint and cadence result in one report-ready figure."""
    daily = daily.reset_index(drop=True)
    footprint = daily["mean_zones_per_wave"].astype(float)
    wave_counts = daily["waves"].astype(float)
    footprint_rolling = footprint.rolling(7, min_periods=7).mean()
    cadence_rolling = wave_counts.rolling(7, min_periods=7).mean()
    early_footprint = float(footprint.iloc[:7].mean())
    late_footprint = float(footprint.iloc[-7:].mean())
    footprint_reduction = 1 - late_footprint / early_footprint
    early_cadence = float(wave_counts.iloc[:7].mean())
    late_cadence = float(wave_counts.iloc[-7:].mean())
    footprint_rho, footprint_p = stats.spearmanr(np.arange(len(daily)), footprint)
    cadence_rho, cadence_p = stats.spearmanr(np.arange(len(daily)), wave_counts)

    width, height = 1050, 770
    left, right = 100, 40
    plot_w = width - left - right
    n = len(daily)
    x_step = plot_w / max(n - 1, 1)
    dates = pd.to_datetime(daily["date"])
    tick_indices = sorted(set([0, 7, 14, 21, 28, 35, n - 1]))

    def x_for(index: int) -> float:
        return left + index * x_step

    def add_panel_grid(
        body: list[str],
        top: float,
        panel_h: float,
        ymax: float,
        ticks: list[float],
        formatter,
    ) -> None:
        for tick in ticks:
            y = top + panel_h * (1 - tick / ymax)
            body.append(
                f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" '
                f'y2="{y:.1f}" stroke="{COLORS["grid"]}"/>'
            )
            body.append(
                text_element(left - 12, y + 4, formatter(tick), 13, COLORS["muted"], "end")
            )

    def add_date_axis(body: list[str], top: float, panel_h: float) -> None:
        bottom = top + panel_h
        for index in tick_indices:
            x = x_for(index)
            date = dates.iloc[index]
            label = f"{date.day} {date.strftime('%b')}"
            body.append(
                f'<line x1="{x:.1f}" y1="{bottom:.1f}" x2="{x:.1f}" '
                f'y2="{bottom+5:.1f}" stroke="{COLORS["muted"]}"/>'
            )
            body.append(
                text_element(x, bottom + 25, label, 13, COLORS["muted"], "middle")
            )
        body.append(
            text_element(
                left + plot_w / 2,
                bottom + 50,
                "Date (2026)",
                13,
                anchor="middle",
                weight=700,
            )
        )

    def add_week_highlights(body: list[str], top: float, panel_h: float) -> None:
        first_width = x_for(6) - left + x_step / 2
        last_x = x_for(n - 7) - x_step / 2
        body.append(
            f'<rect x="{left}" y="{top}" width="{first_width:.1f}" height="{panel_h}" '
            f'fill="#f3f6fa" opacity="0.75"/>'
        )
        body.append(
            f'<rect x="{last_x:.1f}" y="{top}" width="{left+plot_w-last_x:.1f}" '
            f'height="{panel_h}" fill="#fff3f0" opacity="0.75"/>'
        )

    def add_mean_segment(
        body: list[str],
        start_index: int,
        end_index: int,
        value: float,
        top: float,
        panel_h: float,
        ymax: float,
        label: str,
    ) -> None:
        y = top + panel_h * (1 - value / ymax)
        x1, x2 = x_for(start_index), x_for(end_index)
        body.append(
            f'<line x1="{x1:.1f}" y1="{y:.1f}" x2="{x2:.1f}" y2="{y:.1f}" '
            f'stroke="{COLORS["broad"]}" stroke-width="3" stroke-linecap="round"/>'
        )
        body.append(
            text_element((x1 + x2) / 2, y - 8, label, 14, COLORS["broad"], "middle", 700)
        )

    body = [
        text_element(
            45,
            38,
            f"Alert footprint fell {footprint_reduction:.0%} while waves remained frequent",
            30,
            weight=700,
        ),
        text_element(
            45,
            66,
            "Primary Iran-reporting phase, 28 Feb-8 Apr 2026; daily observations and 7-day trailing means",
            16,
            COLORS["muted"],
        ),
    ]

    footprint_top, footprint_h = 125, 220
    footprint_ymax = math.ceil(float(footprint.max()) / 100) * 100
    body.append(
        text_element(left, footprint_top - 18, "A. Mean alert footprint per wave", 18, weight=700)
    )
    body.append(
        text_element(
            left + plot_w,
            footprint_top - 18,
            f"Spearman rho = {footprint_rho:.2f}; p < 0.001"
            if footprint_p < 0.001
            else f"Spearman rho = {footprint_rho:.2f}; p = {footprint_p:.3f}",
            14,
            COLORS["muted"],
            "end",
        )
    )
    add_week_highlights(body, footprint_top, footprint_h)
    add_panel_grid(
        body,
        footprint_top,
        footprint_h,
        footprint_ymax,
        list(np.linspace(0, footprint_ymax, 7)),
        lambda value: f"{value:.0f}",
    )
    raw_points = []
    for i, value in enumerate(footprint):
        x = x_for(i)
        y = footprint_top + footprint_h * (1 - value / footprint_ymax)
        raw_points.append(f"{x:.1f},{y:.1f}")
    body.append(
        f'<polyline points="{" ".join(raw_points)}" fill="none" '
        f'stroke="{COLORS["localized"]}" stroke-width="1.7" opacity="0.55" '
        f'stroke-linejoin="round"/>'
    )
    for i, value in enumerate(footprint):
        x = x_for(i)
        y = footprint_top + footprint_h * (1 - value / footprint_ymax)
        body.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.2" '
            f'fill="{COLORS["localized"]}" opacity="0.72"/>'
        )
    first_y = footprint_top + footprint_h * (1 - footprint.iloc[0] / footprint_ymax)
    body.append(
        text_element(
            x_for(0) + 9,
            first_y + 18,
            f"28 Feb: {footprint.iloc[0]:.0f}",
            13,
            COLORS["muted"],
            weight=700,
        )
    )
    rolling_points = []
    for i, value in enumerate(footprint_rolling):
        if pd.notna(value):
            x = x_for(i)
            y = footprint_top + footprint_h * (1 - value / footprint_ymax)
            rolling_points.append(f"{x:.1f},{y:.1f}")
    body.append(
        f'<polyline points="{" ".join(rolling_points)}" fill="none" '
        f'stroke="{COLORS["accent"]}" stroke-width="4" stroke-linecap="round" '
        f'stroke-linejoin="round"/>'
    )
    add_mean_segment(
        body,
        0,
        6,
        early_footprint,
        footprint_top,
        footprint_h,
        footprint_ymax,
        f"First 7 days: {early_footprint:.1f}",
    )
    add_mean_segment(
        body,
        n - 7,
        n - 1,
        late_footprint,
        footprint_top,
        footprint_h,
        footprint_ymax,
        f"Last 7 days: {late_footprint:.1f}",
    )
    body.append(
        rotated_text_element(
            27,
            footprint_top + footprint_h / 2,
            "Mean alerted localities per wave",
            16,
        )
    )
    add_date_axis(body, footprint_top, footprint_h)

    cadence_top, cadence_h = 445, 170
    cadence_ymax = math.ceil(float(wave_counts.max()) / 10) * 10
    body.append(text_element(left, cadence_top - 18, "B. Reconstructed waves per day", 18, weight=700))
    body.append(
        text_element(
            left + plot_w,
            cadence_top - 18,
            f"Spearman rho = {cadence_rho:+.2f}; p = {cadence_p:.3f}",
            14,
            COLORS["muted"],
            "end",
        )
    )
    add_week_highlights(body, cadence_top, cadence_h)
    add_panel_grid(
        body,
        cadence_top,
        cadence_h,
        cadence_ymax,
        list(np.linspace(0, cadence_ymax, 5)),
        lambda value: f"{value:.0f}",
    )
    bar_w = x_step * 0.58
    for i, value in enumerate(wave_counts):
        x = x_for(i) - bar_w / 2
        y = cadence_top + cadence_h * (1 - value / cadence_ymax)
        body.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" '
            f'height="{cadence_top+cadence_h-y:.1f}" fill="{COLORS["localized"]}" '
            f'opacity="0.48" rx="1.5"/>'
        )
    last_y = cadence_top + cadence_h * (1 - wave_counts.iloc[-1] / cadence_ymax)
    body.append(
        text_element(
            x_for(n - 1) - 7,
            last_y - 8,
            f"8 Apr: {wave_counts.iloc[-1]:.0f}",
            13,
            COLORS["muted"],
            "end",
            700,
        )
    )
    rolling_points = []
    for i, value in enumerate(cadence_rolling):
        if pd.notna(value):
            x = x_for(i)
            y = cadence_top + cadence_h * (1 - value / cadence_ymax)
            rolling_points.append(f"{x:.1f},{y:.1f}")
    body.append(
        f'<polyline points="{" ".join(rolling_points)}" fill="none" '
        f'stroke="{COLORS["accent"]}" stroke-width="4" stroke-linecap="round" '
        f'stroke-linejoin="round"/>'
    )
    add_mean_segment(
        body,
        0,
        6,
        early_cadence,
        cadence_top,
        cadence_h,
        cadence_ymax,
        f"First 7 days: {early_cadence:.1f}",
    )
    add_mean_segment(
        body,
        n - 7,
        n - 1,
        late_cadence,
        cadence_top,
        cadence_h,
        cadence_ymax,
        f"Last 7 days: {late_cadence:.1f}",
    )
    body.append(
        rotated_text_element(27, cadence_top + cadence_h / 2, "Waves per day", 16)
    )
    add_date_axis(body, cadence_top, cadence_h)

    legend_y = 712
    body.append(
        f'<circle cx="{left+5}" cy="{legend_y-4}" r="4" fill="{COLORS["localized"]}" opacity="0.72"/>'
    )
    body.append(text_element(left + 17, legend_y, "Daily value", 14, COLORS["muted"]))
    body.append(
        f'<line x1="{left+125}" y1="{legend_y-4}" x2="{left+153}" y2="{legend_y-4}" '
        f'stroke="{COLORS["accent"]}" stroke-width="4" stroke-linecap="round"/>'
    )
    body.append(text_element(left + 163, legend_y, "7-day trailing mean", 14, COLORS["muted"]))
    body.append(
        f'<line x1="{left+335}" y1="{legend_y-4}" x2="{left+363}" y2="{legend_y-4}" '
        f'stroke="{COLORS["broad"]}" stroke-width="3" stroke-linecap="round"/>'
    )
    body.append(text_element(left + 373, legend_y, "First/last 7-day mean", 14, COLORS["muted"]))
    body.append(
        text_element(
            left,
            748,
            "Footprint is the number of distinct alerted localities, not missile count.",
            13,
            COLORS["muted"],
        )
    )
    write_svg("primary_phase_summary", width, height, body)


def make_persistence_figure(results: pd.DataFrame) -> None:
    width, height = 1050, 545
    left, right, top, bottom = 175, 275, 125, 95
    plot_w, plot_h = width - left - right, height - top - bottom
    ordered = results.set_index("family").loc[["all", "localized", "broad"]].reset_index()
    xmax = math.ceil(
        max(
            ordered["observed_mean_jaccard"].max(),
            ordered["shuffled_mean_jaccard_q975"].max(),
        )
        * 100
    ) / 100 + 0.02

    def x_for(value: float) -> float:
        return left + value / xmax * plot_w

    body = [
        text_element(
            40,
            38,
            "Adjacent waves overlap more than shuffled order predicts",
            30,
            weight=700,
        ),
        text_element(
            40,
            66,
            "Mean Jaccard locality overlap; 2,000 within-day wave-order shuffles",
            16,
            COLORS["muted"],
        ),
    ]
    for tick in np.linspace(0, xmax, 7):
        x = x_for(float(tick))
        body.append(
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+plot_h}" '
            f'stroke="{COLORS["grid"]}"/>'
        )
        body.append(
            text_element(x, top + plot_h + 27, f"{tick:.2f}", 13, COLORS["muted"], "middle")
        )

    group_h = plot_h / len(ordered)
    labels = {"all": "All waves", "broad": "Broad", "localized": "Localized"}
    for i, row in enumerate(ordered.itertuples(index=False)):
        center_y = top + (i + 0.5) * group_h
        lower_x = x_for(row.shuffled_mean_jaccard_q025)
        upper_x = x_for(row.shuffled_mean_jaccard_q975)
        null_x = x_for(row.shuffled_mean_jaccard)
        observed_x = x_for(row.observed_mean_jaccard)
        body.append(
            f'<line x1="{left}" y1="{center_y+group_h/2:.1f}" x2="{left+plot_w}" '
            f'y2="{center_y+group_h/2:.1f}" stroke="{COLORS["grid"]}" opacity="0.7"/>'
        )
        body.append(
            f'<line x1="{lower_x:.1f}" y1="{center_y:.1f}" x2="{upper_x:.1f}" '
            f'y2="{center_y:.1f}" stroke="{COLORS["null"]}" stroke-width="8" '
            f'stroke-linecap="round"/>'
        )
        for endpoint in [lower_x, upper_x]:
            body.append(
                f'<line x1="{endpoint:.1f}" y1="{center_y-9:.1f}" x2="{endpoint:.1f}" '
                f'y2="{center_y+9:.1f}" stroke="{COLORS["muted"]}" stroke-width="1.5"/>'
            )
        body.append(
            f'<rect x="{null_x-5:.1f}" y="{center_y-5:.1f}" width="10" height="10" '
            f'fill="{COLORS["null"]}" stroke="{COLORS["ink"]}" stroke-width="1"/>'
        )
        body.append(
            f'<circle cx="{observed_x:.1f}" cy="{center_y:.1f}" r="7" '
            f'fill="{COLORS["accent"]}" stroke="white" stroke-width="2"/>'
        )
        body.append(
            text_element(left - 18, center_y - 5, labels[row.family], 16, anchor="end", weight=700)
        )
        body.append(
            text_element(
                left - 18,
                center_y + 17,
                f"n={int(row.adjacent_pairs)} adjacent pairs",
                13,
                COLORS["muted"],
                "end",
            )
        )
        p_label = (
            "p<0.001"
            if row.permutation_p_value < 0.001
            else f"p={row.permutation_p_value:.3f}"
        )
        body.append(
            text_element(
                left + plot_w + 18,
                center_y - 12,
                f"Observed {row.observed_mean_jaccard:.3f} | shuffled {row.shuffled_mean_jaccard:.3f}",
                14,
                COLORS["ink"],
                "start",
                700,
            )
        )
        body.append(
            text_element(
                left + plot_w + 18,
                center_y + 10,
                f"{row.observed_to_null_ratio:.2f}x null; {p_label}",
                13,
                COLORS["muted"],
                "start",
            )
        )

    body.append(
        f'<circle cx="{width-400}" cy="90" r="6" fill="{COLORS["accent"]}"/>'
    )
    body.append(text_element(width - 387, 95, "Observed", 14))
    body.append(
        f'<line x1="{width-300}" y1="90" x2="{width-270}" y2="90" '
        f'stroke="{COLORS["null"]}" stroke-width="8" stroke-linecap="round"/>'
    )
    body.append(
        f'<rect x="{width-290}" y="85" width="10" height="10" '
        f'fill="{COLORS["null"]}" stroke="{COLORS["ink"]}" stroke-width="1"/>'
    )
    body.append(text_element(width - 260, 95, "Shuffled mean and 95% interval", 14))
    body.append(
        text_element(
            left + plot_w / 2,
            height - 42,
            "Mean Jaccard locality overlap",
            16,
            anchor="middle",
            weight=700,
        )
    )
    body.append(
        text_element(
            left,
            height - 13,
            "Shuffling preserves each wave's footprint and localities; only within-day order changes.",
            12,
            COLORS["muted"],
        )
    )
    write_svg("target_persistence", width, height, body)


def make_warning_figure(waves: pd.DataFrame) -> None:
    summary = waves.groupby("family").agg(
        mean_coverage=("warning_coverage", "mean"),
        median_coverage=("warning_coverage", "median"),
        waves=("wave_id", "size"),
    )
    width, height = 760, 570
    left, right, top, bottom = 100, 45, 100, 100
    plot_w, plot_h = width - left - right, height - top - bottom
    body = [
        text_element(38, 38, "Broad waves have a distinct pre-warning signature", 24, weight=700),
        text_element(38, 66, "Second war: share of a wave's zones warned in the previous hour", 14, COLORS["muted"]),
    ]
    for tick in np.linspace(0, 1, 6):
        y = top + plot_h * (1 - tick)
        body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="{COLORS["grid"]}"/>')
        body.append(text_element(left - 10, y + 5, f"{tick:.0%}", 12, COLORS["muted"], "end"))
    for i, (family, color) in enumerate([("localized", COLORS["localized"]), ("broad", COLORS["broad"])]):
        row = summary.loc[family]
        center = left + (i + 0.5) * plot_w / 2
        bar_w = 145
        value = row["mean_coverage"]
        y = top + plot_h * (1 - value)
        body.append(f'<rect x="{center-bar_w/2:.1f}" y="{y:.1f}" width="{bar_w}" height="{top+plot_h-y:.1f}" rx="5" fill="{color}" opacity="0.9"/>')
        body.append(text_element(center, y - 12, f"{value:.0%} mean", 16, color, "middle", 700))
        body.append(text_element(center, top + plot_h + 32, family.title(), 15, anchor="middle", weight=700))
        body.append(text_element(center, top + plot_h + 54, f"median {row['median_coverage']:.0%}; n={int(row['waves'])}", 12, COLORS["muted"], "middle"))
    write_svg("prewarning_signature", width, height, body)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    missile_by_war: dict[str, pd.DataFrame] = {}
    diagnostics: dict[str, dict[str, int]] = {}
    wave_rows_by_war: dict[str, pd.DataFrame] = {}
    waves_by_war: dict[str, pd.DataFrame] = {}
    sets_by_war: dict[str, dict[int, set[str]]] = {}

    for war, path in WAR_FILES.items():
        missile, war_diagnostics = load_war(path)
        rows, waves, sets = build_waves(missile, DEFAULT_GAP_MINUTES)
        missile_by_war[war] = missile
        diagnostics[war] = war_diagnostics
        wave_rows_by_war[war] = rows
        waves_by_war[war] = waves
        sets_by_war[war] = sets

    # Select k by relative validation, then learn one common footprint split
    # from the larger second-war dataset.
    cluster_validation = compute_cluster_validation(waves_by_war["second_war"])
    selected_k = int(
        cluster_validation.loc[
            cluster_validation["eligible_for_family_selection"]
        ]
        .dropna(subset=["silhouette_score"])
        .sort_values(["silhouette_score", "k"], ascending=[False, True])
        .iloc[0]["k"]
    )
    if selected_k != 2:
        raise ValueError(
            f"The two-family analysis requires k=2, but validation selected k={selected_k}"
        )
    family_cutoff, cluster_details = discover_family_cutoff(waves_by_war["second_war"])
    k2_validation = cluster_validation.set_index("k").loc[2]
    cluster_details.update(
        {
            "selected_k_by_maximum_silhouette": selected_k,
            "k2_silhouette_score": float(k2_validation["silhouette_score"]),
            "k2_sse_reduction_from_k1": float(
                k2_validation["sse_reduction_from_previous_k"]
            ),
            "k2_mean_pairwise_adjusted_rand": float(
                k2_validation["mean_pairwise_adjusted_rand"]
            ),
            "stability_initializations": len(CLUSTER_STABILITY_SEEDS),
        }
    )
    first_warning_dates: dict[str, pd.Timestamp | None] = {}
    for war in WAR_FILES:
        waves = add_family(waves_by_war[war], family_cutoff)
        waves, first_warning = add_warning_features(
            WAR_FILES[war], wave_rows_by_war[war], waves
        )
        waves["war"] = war
        waves_by_war[war] = waves
        first_warning_dates[war] = first_warning

    second_missile = missile_by_war["second_war"]
    start_date = second_missile["timestamp"].dt.floor("D").min()
    end_date = second_missile["timestamp"].dt.floor("D").max()
    calendar = pd.date_range(start_date, end_date)
    daily_alert_rows = (
        second_missile.groupby(second_missile["timestamp"].dt.floor("D"))
        .size()
        .reindex(calendar, fill_value=0)
    )
    if not SOURCE_EVENTS.exists():
        raise FileNotFoundError(
            f"Missing {SOURCE_EVENTS}. Run source_labeling/run_pipeline.py --offline first."
        )
    source_events = pd.read_csv(SOURCE_EVENTS)
    phase_start, phase_end_exclusive, source_reporting_days = (
        longest_continuous_source_phase(source_events, source="iran")
    )
    phase_start_day = int((phase_start - start_date).days)
    phase_end_day = int((phase_end_exclusive - start_date).days)
    if phase_start_day < 0 or phase_end_day > len(calendar):
        raise ValueError("Official-source phase falls outside the alert dataset calendar")

    # These change points are diagnostics only. The primary phase is defined
    # independently from official source-report dates, not from alert footprint.
    alert_row_change_points = find_three_regimes(daily_alert_rows)

    second_daily = daily_metrics(waves_by_war["second_war"], len(calendar))
    second_daily["missile_alert_rows"] = daily_alert_rows.to_numpy()
    second_daily["date"] = calendar.date.astype(str)
    high_daily = second_daily.loc[
        (second_daily["day_index"] >= phase_start_day)
        & (second_daily["day_index"] < phase_end_day)
    ].copy()
    daily_wave_counts = second_daily["waves"].fillna(0)
    wave_count_change_points = find_three_regimes(daily_wave_counts)

    persistence = persistence_permutation_test(
        waves_by_war["second_war"], sets_by_war["second_war"], phase_end_day
    )
    sensitivity = compute_sensitivity(second_missile, phase_end_day)
    persistence_sensitivity = compute_persistence_gap_sensitivity(
        second_missile, phase_end_day
    )
    phase_sensitivity = compute_phase_boundary_sensitivity(
        waves_by_war["second_war"], start_date, phase_end_day
    )
    opening_day_sensitivity = compute_phase_start_sensitivity(high_daily)
    reality_checks = make_wave_reality_checks(
        waves_by_war["second_war"], sets_by_war["second_war"], family_cutoff
    )
    if not LABELED_WAVES.exists():
        raise FileNotFoundError(
            f"Missing {LABELED_WAVES}. Run source_labeling/run_pipeline.py --offline first."
        )
    source_trend_diagnostics = compute_source_trend_diagnostics(
        pd.read_csv(LABELED_WAVES), phase_start, phase_end_exclusive
    )

    # Main trend statistics use days as units to avoid treating locality rows as independent.
    footprint_rho, footprint_p = stats.spearmanr(
        high_daily["day_index"], high_daily["mean_zones_per_wave"]
    )
    cadence_rho, cadence_p = stats.spearmanr(
        high_daily["day_index"], high_daily["waves"]
    )
    broad_fraction_rho, broad_fraction_p = stats.spearmanr(
        high_daily["day_index"], high_daily["broad_fraction"]
    )
    broad_daily = high_daily.dropna(subset=["median_broad_wave_zones"])
    broad_size_rho, broad_size_p = stats.spearmanr(
        broad_daily["day_index"], broad_daily["median_broad_wave_zones"]
    )
    early = high_daily.iloc[:7]
    late = high_daily.iloc[-7:]
    early_late_test = stats.mannwhitneyu(
        early["mean_zones_per_wave"],
        late["mean_zones_per_wave"],
        alternative="greater",
    )

    # Comparable first 12 calendar days in each dataset.
    comparison_records = []
    for war, waves in waves_by_war.items():
        first_12 = waves.loc[waves["day_index"] <= 11]
        broad = first_12.loc[first_12["family"] == "broad"]
        comparison_records.append(
            {
                "war": war,
                "waves_first_12_days": len(first_12),
                "waves_per_day": len(first_12) / 12,
                "median_zones_per_wave": first_12["zone_count"].median(),
                "mean_zones_per_wave": first_12["zone_count"].mean(),
                "broad_wave_fraction": (first_12["family"] == "broad").mean(),
                "median_broad_wave_zones": broad["zone_count"].median(),
            }
        )
    comparison = pd.DataFrame(comparison_records)

    war_summary_records = []
    for war, waves in waves_by_war.items():
        row = {
            "war": war,
            **diagnostics[war],
            "reconstructed_waves": len(waves),
            "median_zones_per_wave": waves["zone_count"].median(),
            "mean_zones_per_wave": waves["zone_count"].mean(),
            "broad_wave_fraction": (waves["family"] == "broad").mean(),
            "first_preliminary_warning": (
                None
                if first_warning_dates[war] is None
                else first_warning_dates[war].isoformat()
            ),
        }
        war_summary_records.append(row)
    war_summary = pd.DataFrame(war_summary_records)

    warning_summary = (
        waves_by_war["second_war"]
        .groupby("family")
        .agg(
            waves=("wave_id", "size"),
            mean_zone_warning_coverage=("warning_coverage", "mean"),
            median_zone_warning_coverage=("warning_coverage", "median"),
            share_of_waves_with_any_matched_warning=(
                "median_warning_lead_minutes",
                lambda x: x.notna().mean(),
            ),
            median_lead_minutes_when_matched=("median_warning_lead_minutes", "median"),
        )
        .reset_index()
    )

    key_findings = pd.DataFrame(
        [
            ["wave_gap_minutes", DEFAULT_GAP_MINUTES],
            ["broad_family_cutoff_zones", family_cutoff],
            ["cluster_validation_selected_k", selected_k],
            ["k2_silhouette_score", k2_validation["silhouette_score"]],
            [
                "k2_sse_reduction_from_k1",
                k2_validation["sse_reduction_from_previous_k"],
            ],
            [
                "k2_mean_pairwise_adjusted_rand",
                k2_validation["mean_pairwise_adjusted_rand"],
            ],
            ["primary_phase_start_date", phase_start.date().isoformat()],
            [
                "primary_phase_end_date_inclusive",
                (phase_end_exclusive - pd.Timedelta(days=1)).date().isoformat(),
            ],
            ["primary_phase_days", phase_end_day - phase_start_day],
            ["official_iran_reporting_days_in_phase", source_reporting_days],
            ["primary_phase_day_vs_mean_footprint_spearman_rho", footprint_rho],
            ["primary_phase_day_vs_mean_footprint_p", footprint_p],
            ["primary_phase_day_vs_daily_wave_count_spearman_rho", cadence_rho],
            ["primary_phase_day_vs_daily_wave_count_p", cadence_p],
            ["primary_phase_day_vs_broad_fraction_spearman_rho", broad_fraction_rho],
            ["primary_phase_day_vs_broad_fraction_p", broad_fraction_p],
            ["primary_phase_day_vs_daily_broad_median_spearman_rho", broad_size_rho],
            ["primary_phase_day_vs_daily_broad_median_p", broad_size_p],
            ["early_7_days_mean_zones_per_wave", early["mean_zones_per_wave"].mean()],
            ["late_7_days_mean_zones_per_wave", late["mean_zones_per_wave"].mean()],
            [
                "early_to_late_footprint_reduction",
                1
                - late["mean_zones_per_wave"].mean()
                / early["mean_zones_per_wave"].mean(),
            ],
            ["early_7_days_mean_waves_per_day", early["waves"].mean()],
            ["late_7_days_mean_waves_per_day", late["waves"].mean()],
            ["early_vs_late_footprint_mann_whitney_p", early_late_test.pvalue],
            [
                "all_adjacent_waves_observed_to_shuffled_overlap_ratio",
                persistence.set_index("family").loc["all", "observed_to_null_ratio"],
            ],
            [
                "all_adjacent_waves_permutation_p",
                persistence.set_index("family").loc["all", "permutation_p_value"],
            ],
        ],
        columns=["metric", "value"],
    )

    all_waves = pd.concat(waves_by_war.values(), ignore_index=True)
    columns = [
        "war",
        "wave_id",
        "start",
        "end",
        "date",
        "day_index",
        "hour",
        "duration_minutes",
        "gap_from_previous_hours",
        "alert_rows",
        "zone_count",
        "burst_count",
        "family",
        "warning_coverage",
        "median_warning_lead_minutes",
    ]

    all_waves[columns].to_csv(TABLES / "wave_metrics.csv", index=False)
    second_daily.to_csv(TABLES / "second_war_daily_metrics.csv", index=False)
    war_summary.to_csv(TABLES / "war_summary.csv", index=False)
    comparison.to_csv(TABLES / "first_12_days_comparison.csv", index=False)
    cluster_validation.to_csv(TABLES / "cluster_validation.csv", index=False)
    warning_summary.to_csv(TABLES / "warning_summary.csv", index=False)
    persistence.to_csv(TABLES / "target_persistence_test.csv", index=False)
    persistence_sensitivity.to_csv(
        TABLES / "target_persistence_gap_sensitivity.csv", index=False
    )
    sensitivity.to_csv(TABLES / "wave_gap_sensitivity.csv", index=False)
    phase_sensitivity.to_csv(TABLES / "phase_boundary_sensitivity.csv", index=False)
    opening_day_sensitivity.to_csv(
        TABLES / "opening_day_sensitivity.csv", index=False
    )
    reality_checks.to_csv(TABLES / "wave_reality_checks.csv", index=False)
    source_trend_diagnostics.to_csv(
        TABLES / "source_trend_diagnostics.csv", index=False
    )
    key_findings.to_csv(TABLES / "key_findings.csv", index=False, quoting=csv.QUOTE_MINIMAL)

    metadata = {
        "wave_gap_minutes": DEFAULT_GAP_MINUTES,
        "gap_sensitivity_minutes": GAP_SENSITIVITY,
        "family_cutoff_zones": family_cutoff,
        "family_cluster_details": cluster_details,
        "cluster_validation": {
            "candidate_k_values": CLUSTER_K_VALUES,
            "selection_rule": "maximum silhouette among solutions whose smallest cluster contains at least 5% of waves; SSE elbow and seed stability are corroborating diagnostics",
            "selected_k": selected_k,
            "minimum_cluster_fraction": MIN_CLUSTER_FRACTION,
            "stability_initializations": len(CLUSTER_STABILITY_SEEDS),
        },
        "primary_phase": {
            "definition": "longest continuous run of days with eligible official Iran incoming-threat reports",
            "start_date": phase_start.date().isoformat(),
            "end_date_inclusive": (
                phase_end_exclusive - pd.Timedelta(days=1)
            ).date().isoformat(),
            "days": source_reporting_days,
        },
        "diagnostic_change_points_not_used_to_define_primary_phase": {
            "daily_alert_rows_day_indices": list(alert_row_change_points),
            "daily_wave_counts_day_indices": list(wave_count_change_points),
        },
        "phase_end_sensitivity_days": PHASE_END_SENSITIVITY_DAYS,
        "opening_day_exclusions": PHASE_START_TRIM_DAYS,
        "source_trend_diagnostics": source_trend_diagnostics.to_dict(
            orient="records"
        ),
        "random_seed": RANDOM_SEED,
        "permutations": N_PERMUTATIONS,
        "persistence_sensitivity_permutations": N_SENSITIVITY_PERMUTATIONS,
    }
    (TABLES / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    make_distribution_figure(waves_by_war, family_cutoff, cluster_validation)
    make_timeline_figure(waves_by_war["second_war"], phase_end_day, family_cutoff)
    make_primary_phase_summary_figure(high_daily)
    make_evolution_figure(high_daily)
    make_persistence_figure(persistence)
    make_warning_figure(waves_by_war["second_war"])

    print(f"Wrote investigation outputs to {HERE}")
    print(key_findings.to_string(index=False))


if __name__ == "__main__":
    main()

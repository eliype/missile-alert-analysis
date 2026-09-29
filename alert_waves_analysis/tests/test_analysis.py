from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


ANALYSIS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ANALYSIS_DIR))

from analysis import (  # noqa: E402
    build_waves,
    compute_cluster_validation,
    compute_phase_start_sensitivity,
    longest_continuous_source_phase,
    persistence_permutation_test,
)


class WaveConstructionTests(unittest.TestCase):
    def test_gap_rule_starts_a_new_wave_only_after_threshold(self) -> None:
        missile = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [
                        "2026-03-01 10:00",
                        "2026-03-01 10:08",
                        "2026-03-01 10:17",
                    ]
                ),
                "data": ["Alpha", "Beta", "Gamma"],
            }
        )
        rows, waves, zone_sets = build_waves(missile, gap_minutes=8)
        self.assertEqual(rows["wave_id"].tolist(), [0, 0, 1])
        self.assertEqual(waves["zone_count"].tolist(), [2, 1])
        self.assertEqual(zone_sets[0], {"Alpha", "Beta"})


class PhaseDefinitionTests(unittest.TestCase):
    def test_longest_continuous_official_source_run_is_selected(self) -> None:
        events = pd.DataFrame(
            {
                "reported_source": ["iran"] * 5 + ["lebanon"],
                "eligible_for_missile_wave_match": [True] * 6,
                "event_time_local": [
                    "2026-03-01T10:00:00+02:00",
                    "2026-03-02T10:00:00+02:00",
                    "2026-03-03T10:00:00+02:00",
                    "2026-03-10T10:00:00+02:00",
                    "2026-03-11T10:00:00+02:00",
                    "2026-03-04T10:00:00+02:00",
                ],
            }
        )
        start, end_exclusive, days = longest_continuous_source_phase(events)
        self.assertEqual(start, pd.Timestamp("2026-03-01"))
        self.assertEqual(end_exclusive, pd.Timestamp("2026-03-04"))
        self.assertEqual(days, 3)

    def test_opening_day_sensitivity_recomputes_the_trend(self) -> None:
        daily = pd.DataFrame(
            {
                "day_index": range(10),
                "date": pd.date_range("2026-03-01", periods=10).date.astype(str),
                "mean_zones_per_wave": range(10, 0, -1),
            }
        )
        result = compute_phase_start_sensitivity(daily, trim_days=(0, 3))
        self.assertEqual(result["remaining_days"].tolist(), [10, 7])
        self.assertEqual(result["analysis_start_date"].tolist()[1], "2026-03-04")
        for rho in result["day_vs_daily_mean_footprint_spearman_rho"]:
            self.assertAlmostEqual(rho, -1.0)


class PersistenceTests(unittest.TestCase):
    def test_family_pairs_must_be_adjacent_in_complete_sequence(self) -> None:
        waves = pd.DataFrame(
            {
                "wave_id": [1, 2, 3, 4],
                "day_index": [0, 0, 0, 0],
                "start": pd.to_datetime(
                    [
                        "2026-03-01 10:00",
                        "2026-03-01 11:00",
                        "2026-03-01 12:00",
                        "2026-03-01 13:00",
                    ]
                ),
                "family": ["broad", "broad", "localized", "localized"],
            }
        )
        zone_sets = {
            1: {"A", "B"},
            2: {"A", "B"},
            3: {"Y", "Z"},
            4: {"Y", "Z"},
        }
        result = persistence_permutation_test(
            waves, zone_sets, high_intensity_end=1, n_permutations=50
        ).set_index("family")
        self.assertEqual(result.loc["all", "adjacent_pairs"], 3)
        self.assertEqual(result.loc["broad", "adjacent_pairs"], 1)
        self.assertEqual(result.loc["localized", "adjacent_pairs"], 1)
        self.assertEqual(result.loc["broad", "observed_mean_jaccard"], 1.0)
        self.assertEqual(result.loc["localized", "observed_mean_jaccard"], 1.0)


class ClusterValidationTests(unittest.TestCase):
    def test_relative_validation_recovers_a_clear_two_group_split(self) -> None:
        waves = pd.DataFrame(
            {
                "zone_count": [1, 2, 2, 3, 4, 90, 110, 140, 180, 230],
            }
        )
        result = compute_cluster_validation(
            waves, k_values=(1, 2, 3), stability_seeds=tuple(range(5))
        ).set_index("k")
        self.assertLess(result.loc[2, "sse_relative_to_k1"], 0.2)
        self.assertGreater(
            result.loc[2, "silhouette_score"], result.loc[3, "silhouette_score"]
        )
        self.assertGreater(result.loc[2, "mean_pairwise_adjusted_rand"], 0.9)


if __name__ == "__main__":
    unittest.main()

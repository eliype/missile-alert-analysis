from __future__ import annotations

import sys
import unittest
from datetime import datetime
from pathlib import Path


SOURCE_LABELING_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_LABELING_DIR))

from event_rules import (  # noqa: E402
    classify_source,
    classify_threat,
    classify_time_basis,
    is_relevant_incoming_threat,
    referenced_siren_times,
)
from idf_telegram import parse_page  # noqa: E402
from wave_matching import make_wave_localities  # noqa: E402


class EventRuleTests(unittest.TestCase):
    def test_iran_detection(self) -> None:
        text = (
            "IDF: A short while ago, the IDF identified missiles launched from Iran "
            "toward the territory of the State of Israel."
        )
        self.assertEqual(classify_source(text), ("iran", "high"))
        self.assertEqual(classify_threat(text), "missile")
        self.assertEqual(classify_time_basis(text), "detection_before_alert")
        self.assertTrue(is_relevant_incoming_threat(text, "iran"))

    def test_lebanon_confirmation(self) -> None:
        text = (
            "Following the sirens that sounded in northern Israel, several projectiles "
            "that crossed from Lebanon into Israeli territory were identified."
        )
        self.assertEqual(classify_source(text), ("lebanon", "high"))
        self.assertEqual(classify_threat(text), "projectile")
        self.assertEqual(classify_time_basis(text), "confirmation_after_alert")

    def test_idf_strike_is_not_incoming_threat(self) -> None:
        text = "The IDF struck several Hezbollah rocket launchers in southern Lebanon."
        source, _ = classify_source(text)
        self.assertEqual(source, "unknown")
        self.assertFalse(is_relevant_incoming_threat(text, source))

    def test_uav_is_detected_but_not_missile(self) -> None:
        text = "A UAV that crossed from Lebanon into Israeli territory was intercepted."
        self.assertEqual(classify_source(text), ("lebanon", "high"))
        self.assertEqual(classify_threat(text), "uav")

    def test_retrospective_post_is_not_timed_as_current_event(self) -> None:
        text = (
            "The IDF identified a Hezbollah launcher as it launched rockets toward "
            "the State of Israel yesterday."
        )
        source, _ = classify_source(text)
        self.assertEqual(source, "lebanon")
        self.assertFalse(is_relevant_incoming_threat(text, source))

    def test_prevented_launch_is_not_an_incoming_threat(self) -> None:
        text = (
            "The IDF continues to thwart missile launches from Iran toward Israel and "
            "prevented the launch by striking the launcher."
        )
        source, _ = classify_source(text)
        self.assertEqual(source, "iran")
        self.assertFalse(is_relevant_incoming_threat(text, source))

    def test_explicit_siren_time(self) -> None:
        text = "Following the sirens that sounded in northern Israel at 15:00, a launch from Lebanon was intercepted."
        post_time = datetime.fromisoformat("2026-03-04T15:12:00+02:00")
        values = referenced_siren_times(text, post_time)
        self.assertEqual(values[0].hour, 15)
        self.assertEqual(values[0].minute, 0)


class TelegramParserTests(unittest.TestCase):
    def test_message_text_and_timestamp(self) -> None:
        fixture = """
        <html><head><link rel="prev" href="/s/idfofficial?before=123"></head>
        <body><div class="js-widget_message" data-post="idfofficial/123">
        <div class="tgme_widget_message_text js-message_text">Line one<br/>Line two</div>
        <time datetime="2026-03-04T13:12:00+00:00">13:12</time></div></body></html>
        """
        messages, cursor = parse_page(fixture)
        self.assertEqual(cursor, 123)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].post_id, 123)
        self.assertEqual(messages[0].text, "Line one Line two")


class WaveLocalityTests(unittest.TestCase):
    def test_wave_localities_preserve_footprint_and_counts(self) -> None:
        import pandas as pd

        missile_rows = pd.DataFrame(
            {
                "wave_id": [3, 3, 3, 4],
                "data": ["Alpha", "Alpha", "Beta", "Gamma"],
                "timestamp": pd.to_datetime(
                    [
                        "2026-03-01 10:00",
                        "2026-03-01 10:02",
                        "2026-03-01 10:01",
                        "2026-03-01 11:00",
                    ]
                ),
            }
        )
        waves = pd.DataFrame(
            {
                "wave_id": [3, 4],
                "wave_start": pd.to_datetime(["2026-03-01 10:00", "2026-03-01 11:00"]),
                "wave_end": pd.to_datetime(["2026-03-01 10:02", "2026-03-01 11:00"]),
                "zone_count": [2, 1],
                "wave_family_proxy": ["localized", "localized"],
                "reported_source": ["iran", "unknown"],
                "source_label_confidence": ["high", "none"],
            }
        )
        result = make_wave_localities(missile_rows, waves)
        alpha = result.loc[result["locality"] == "Alpha"].iloc[0]
        self.assertEqual(len(result), 3)
        self.assertEqual(alpha["alert_count"], 2)
        self.assertEqual(alpha["first_alert_time"], pd.Timestamp("2026-03-01 10:00"))
        self.assertEqual(alpha["reported_source"], "iran")


if __name__ == "__main__":
    unittest.main()

# Alert source labeling

This directory builds a processed alert-wave dataset with probable launch sources from official IDF reports.

It addresses the hypothesis:

> Are broad alert waves mainly Iranian launches, while localized waves mainly come from Lebanon?

The output is useful, but intentionally conservative: unmatched or ambiguous waves remain `unknown`.

## Data sources

- Alerts: `data/processed/second_war.csv`
- Official reports: the IDF's public Telegram channel, `https://t.me/s/idfofficial`

The main IDF website displays an anti-bot page to ordinary automated requests. The public Telegram channel is also official, provides stable post URLs and UTC timestamps, and is therefore the reproducible source used by this pipeline. Every extracted event and wave label retains its official source URL.

## Quick start

From the repository root:

```bash
# Full run: download/cache official posts and rebuild all outputs
.venv/bin/python alert_waves_analysis/source_labeling/run_pipeline.py

# Fast reproducible rerun from the saved official-message snapshot
.venv/bin/python alert_waves_analysis/source_labeling/run_pipeline.py --offline

# Rule and parser tests
.venv/bin/python -m unittest discover \
  -s alert_waves_analysis/source_labeling/tests -v
```

The default date range is 2026-02-28 through 2026-06-17. Use `--help` for date, wave-gap, refresh, and input options.

## Pipeline

1. **Download official posts**
   - Crawls the public IDF Telegram pages backward.
   - Converts UTC post timestamps to `Asia/Jerusalem`, including daylight-saving changes.
   - Caches HTML pages under `alert_waves_analysis/data/raw/idf_telegram_pages/`.
   - Saves the compact source snapshot as `alert_waves_analysis/data/raw/idf_telegram_messages.csv`.

2. **Extract incoming threats**
   - Transparent regex rules identify explicit origins such as “launched from Iran” or “crossed from Lebanon.”
   - Threat type is recorded separately: missile, rocket, projectile, UAV, mixed, or unknown.
   - Retrospective summaries and reports about prevented launches are excluded from automatic timing.
   - Explicit siren times inside a post are extracted when available.

3. **Reconstruct alert waves**
   - Keeps only missile-alert rows (`category == 1`).
   - Removes duplicate locality/timestamp rows.
   - Joins consecutive alerts when their gap is no more than 8 minutes.

4. **Match official events to waves**
   - Matching uses only official timing and a time-basis-specific window.
   - It does **not** use footprint size, broad/localized family, pre-warning coverage, or geography. This avoids circularly proving the hypothesis.
   - Each candidate receives `high`, `medium`, `ambiguous`, `unmatched`, or `not_eligible` status.

5. **Aggregate evidence**
   - Consistent high/medium reports become an Iran or Lebanon wave label.
   - Conflicting accepted reports produce `mixed`.
   - All other waves remain `unknown`.

## Investigation datasets

`alert_waves_analysis/data/processed/labeled_alert_waves.csv` contains one row per reconstructed wave.

`alert_waves_analysis/data/processed/wave_localities.csv` contains the corresponding geographical
footprint in long format: one row per `(wave_id, locality)`. It preserves the
first and last alert time for that locality, its number of alert rows, and the
wave-level source label and family. This is the recommended file for mapping,
repeat-locality, and regional analyses; it avoids packing a variable-length
locality list into one CSV cell.

Important columns:

| Column | Meaning |
|---|---|
| `wave_id` | Deterministic wave identifier for this run |
| `wave_start`, `wave_end` | Local alert timestamps |
| `zone_count` | Distinct alerted localities |
| `wave_family_proxy` | `broad` if at least 26 localities; otherwise `localized` |
| `reported_source` | `iran`, `lebanon`, `mixed`, or `unknown` |
| `source_label_confidence` | `high`, `medium`, `conflict`, `manual`, or `none` |
| `source_evidence_count` | Accepted official reports supporting the label |
| `source_event_ids`, `source_post_ids` | Traceable evidence identifiers |
| `source_urls` | Official Telegram post links |

## First-pass results

The automatic pipeline labels 345 of 1,196 waves (28.8%):

- 294 Iran
- 49 Lebanon
- 2 mixed
- 851 unknown

Among independently time-labeled waves:

| Relationship | Result |
|---|---:|
| Iranian waves that are broad | 258 / 294 = **87.8%** |
| Lebanese waves that are localized | 44 / 49 = **89.8%** |
| Broad labeled waves that are Iranian | 258 / 263 = **98.1%** |
| Localized labeled waves that are Lebanese | 44 / 80 = **55.0%** |

The distinction matters: **broad is a strong indicator of Iran, but localized does not automatically mean Lebanon**. Iran also produced 36 independently labeled localized waves.

The Iran-versus-Lebanon odds ratio for a broad wave is 63.1 (Fisher exact p = 9.8e-28). This is an exploratory association and does not correct for selection bias in which events receive official posts.

During the 40-day primary phase, the combined accepted Iran-labeled subset also declines
(daily mean footprint: Spearman rho = -0.507, p = 0.00084). This does **not** provide clean
independent confirmation: the high-confidence subset alone has rho = -0.132 (p = 0.44),
while the medium-confidence subset has rho = -0.425 (p = 0.0070). The source labels are
therefore contextual diagnostics, not evidence needed for the main temporal result.

## Manual validation workflow

`alert_waves_analysis/source_labeling/outputs/validation_sample.csv` contains deterministic samples of high, medium, ambiguous, and unmatched cases. Its review columns remain blank because an independent human review has not yet been completed.

Recommended procedure:

1. Two group members independently inspect each sampled official post and nearby alert time.
2. Record agreement in the `manual_*` columns.
3. Put corrections in `alert_waves_analysis/source_labeling/manual/event_overrides.csv`.
4. Rerun with `--offline`.
5. Report accuracy separately for high- and medium-confidence matches.

Until this review is completed, do not treat the automatic source labels as ground truth or use
them to strengthen the report's core claim. The main analysis uses the dates of eligible official
Iran reports only to define the phase boundary, then measures the trend over all reconstructed waves.

The overrides file is deliberately small and human-readable:

```text
event_id,override_wave_id,override_source,review_status,note
```

## Outputs

- `alert_waves_analysis/data/raw/idf_telegram_messages.csv` — downloaded official-message snapshot
- `alert_waves_analysis/data/processed/idf_incoming_threats.csv` — parsed incoming threats
- `alert_waves_analysis/data/processed/labeled_alert_waves.csv` — one-row-per-wave dataset
- `alert_waves_analysis/data/processed/wave_localities.csv` — one-row-per-locality dataset
- `alert_waves_analysis/source_labeling/outputs/event_wave_matches.csv` — full match evidence and diagnostics
- `alert_waves_analysis/source_labeling/outputs/validation_sample.csv` — manual QA sample
- `alert_waves_analysis/source_labeling/outputs/source_by_wave_family.csv` — source/family contingency table
- `alert_waves_analysis/source_labeling/outputs/source_family_association.csv` — conditional rates, odds ratio, and test
- `alert_waves_analysis/source_labeling/outputs/source_wave_summary.csv` — source-level wave statistics
- `alert_waves_analysis/source_labeling/outputs/iran_labeled_daily_evolution.csv` — Iran-only trend data
- `alert_waves_analysis/source_labeling/outputs/pipeline_summary.json` — parameters, exclusions, and counts

The scraper and event parser live beside the pipeline:

- `alert_waves_analysis/source_labeling/idf_telegram.py` — Telegram crawler and HTML parser
- `alert_waves_analysis/source_labeling/event_rules.py` — transparent source, threat, and event-time rules

## Limitations

- Official posts do not cover every alert wave, so unlabeled waves are not a random sample.
- Post time is sometimes before the siren and sometimes after it; the pipeline models this explicitly, but mistakes remain possible.
- One official post can summarize several launches. Explicit siren times are separated, while less precise summaries may remain ambiguous.
- Some missile-category alerts are triggered by interception debris from UAV events. Threat type is retained so these cases can be filtered.
- `broad` and `localized` remain data-derived proxies, not official military categories.
- A manual accuracy evaluation is required before using labels as final ground truth in the report.

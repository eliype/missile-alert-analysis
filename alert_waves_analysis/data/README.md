# Alert-wave datasets

This directory contains data produced specifically for the alert-wave analysis.
The original war-level inputs remain in the repository-level `data/processed/`
directory because other project analyses also use them.

## Processed data

- `idf_incoming_threats.csv` — official IDF reports parsed into candidate
  incoming-threat events with source, threat type, timing basis, and source URL
- `labeled_alert_waves.csv` — one row per reconstructed alert wave, including
  timing, footprint size, broad/localized family, and cautious source label
- `wave_localities.csv` — one row per locality in each wave, suitable for
  geographic, overlap, and target-persistence analysis

## Raw snapshot

- `raw/idf_telegram_messages.csv` — saved official-message snapshot used by the
  offline source-labeling pipeline

The complete provenance, matching method, limitations, and regeneration
commands are documented in `../source_labeling/README.md`.

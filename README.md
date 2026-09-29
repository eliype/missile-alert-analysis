# Missiles and data: A data-driven look into the patterns of Iran's missile firing during the war

## 1. `CommunityAlerts`
Analyzes Home Front Command alert data to measure warning effectiveness and detect spatial clusters of cities that experience simultaneous alerts.
Uses a 2-minute sliding window to build a network graph of co-occurring alerts. It groups cities into communities using the Girvan-Newman algorithm and exports interactive HTML plots and a CSV mapping.

The analysis was tested with Python 3.13. From the repository root, run:

pip install pandas networkx plotly

From the repository root (or module directory), run:

python CommunityAlerts/communityAlerts.py


## 2. `alert_waves_analysis`

This part of the project asks how missile-alert patterns changed over time. It reconstructs an
alert wave by joining consecutive direct-missile alerts when the gap between them is no more than
eight minutes. It then examines:

- whether waves form distinct broad and localized footprint groups;
- whether their geographic footprint changed during the war;
- whether consecutive waves returned to the same localities; and
- how the wave groups relate to source labels derived from official IDF reports.

A wave's footprint is the number of distinct alerted localities. It is not a missile count,
physical area, or estimate of the exposed population.

### Reproduce the analysis

The analysis was tested with Python 3.12. From the repository root on macOS or Linux, run:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python alert_waves_analysis/run_all.py --test
```

The command runs offline using the committed alert data and official-message snapshot. It
reconstructs the waves, rebuilds the source labels, regenerates the result tables and figures,
and runs both test suites. Use `--online` only if you intentionally want to refresh the official
message source.

Useful locations:

- [`alert_waves_analysis/run_all.py`](alert_waves_analysis/run_all.py) - complete entry point
- [`alert_waves_analysis/README.md`](alert_waves_analysis/README.md) - detailed method, findings, and limitations
- [`alert_waves_analysis/source_labeling/`](alert_waves_analysis/source_labeling/) - official-report parsing and wave matching
- [`alert_waves_analysis/outputs/figures/`](alert_waves_analysis/outputs/figures/) - generated figures
- [`alert_waves_analysis/outputs/tables/`](alert_waves_analysis/outputs/tables/) - generated results and diagnostics

## 3. `trump_correlations`

## Trump Truths & Missile Alerts Analysis Module

This module runs Representational Similarity Analysis (RSA) and the 
Mantel Permutation Test to correlate Trump's Truth Social posts with 
physical rocket alarm time-series data.

### Expected Data Layout
This module relies on shared project data located in the root `data/` directory:
- Input alarms log: `data/processed/second_war.csv`

Generated outputs (truth vectors and rocket-only filtered logs) will be saved 
under `data/` automatically.

### Running the Module

From the repository root (or module directory), run:

```bash
python trump_correlations/run_pipeline.py
"# missile-alert-analysis" 

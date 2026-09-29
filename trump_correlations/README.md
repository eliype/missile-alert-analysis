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
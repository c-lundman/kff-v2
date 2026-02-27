# Lossy Synthetic Datasets

This directory stores corrupted measurement variants derived from perfect ground-truth scenarios.

## Current Scenario

- `day_2026-01-15/`

Each variant folder contains:

- `measured_events.csv`: direction-tagged measured events (includes spurious markers).
- `minute_flows.csv`: 1-minute in/out counts with naive occupancy from raw measured flows.
- `summary.json`: corruption configuration and diagnostic metrics (including negative occupancy minutes).

## Generation

Run:

- `python3 scripts/generate_lossy_datasets.py`

This script requires the perfect scenario at `data/synthetic/perfect/day_2026-01-15/events.csv`.

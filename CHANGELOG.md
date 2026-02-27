# Changelog

## 0.1.0 - 2026-02-27

Initial public release.

### Added

- QP-based queue flow reconciliation (`cvxpy` + `OSQP`).
- Episode detection and per-episode reconciliation pipeline.
- Public DataFrame interface:
  - `estimate_queue_from_timestamps(...)`
  - default output with `Tid` index and columns:
    - `Pax i kö`
    - `Pax ur kö`
    - `Pax in i kö`
    - `Väntetid`
- FIFO wait-time reconstruction from corrected minute flows.
- Synthetic dataset generators:
  - perfect and lossy daily datasets
  - banked multi-day datasets
- Plotting utilities for flow, occupancy, and waiting-time comparison.
- Reusable metrics module for occupancy, wait, physicality, and correction size KPIs.
- Test suite covering core logic, edge cases, and regression checks.
- GitHub Actions CI workflow (`ruff` + `pytest`).


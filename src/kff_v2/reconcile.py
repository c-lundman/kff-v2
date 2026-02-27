"""QP-based reconciliation of noisy minute-binned queue flows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cvxpy as cp
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ReconcileConfig:
    """Configuration for QP flow reconciliation."""

    q0: float = 0.0
    w_in: float = 1.0
    w_out: float = 4.0
    smooth_in: float = 0.0
    smooth_out: float = 0.0
    nonnegative_flows: bool = True
    solver: str = "OSQP"
    eps_abs: float = 1e-5
    eps_rel: float = 1e-5
    max_iter: int = 50_000


def reconcile_minute_flows(
    df: pd.DataFrame,
    config: Optional[ReconcileConfig] = None,
) -> pd.DataFrame:
    """Reconcile noisy inflow/outflow series into physically feasible flows.

    Parameters
    ----------
    df:
        Input DataFrame with columns `minute_start_utc`, `in_count`, `out_count`.
    config:
        Optional reconciliation settings.

    Returns
    -------
    pandas.DataFrame
        Original timeline plus:
        `in_count_measured`, `out_count_measured`,
        `in_count_corrected`, `out_count_corrected`,
        `occupancy_corrected_end`.
    """
    cfg = config or ReconcileConfig()
    required_cols = {"minute_start_utc", "in_count", "out_count"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    work = df.copy()
    work = work.sort_values("minute_start_utc").reset_index(drop=True)

    in_measured = work["in_count"].astype(float).to_numpy()
    out_measured = work["out_count"].astype(float).to_numpy()
    n = len(work)
    if n == 0:
        return pd.DataFrame(
            columns=[
                "minute_start_utc",
                "in_count_measured",
                "out_count_measured",
                "in_count_corrected",
                "out_count_corrected",
                "occupancy_corrected_end",
            ]
        )

    i = cp.Variable(n)
    o = cp.Variable(n)
    q = cp.Variable(n)

    objective_terms = [
        cfg.w_in * cp.sum_squares(i - in_measured),
        cfg.w_out * cp.sum_squares(o - out_measured),
    ]
    if cfg.smooth_in > 0 and n > 1:
        objective_terms.append(cfg.smooth_in * cp.sum_squares(i[1:] - i[:-1]))
    if cfg.smooth_out > 0 and n > 1:
        objective_terms.append(cfg.smooth_out * cp.sum_squares(o[1:] - o[:-1]))

    constraints = [
        q[0] == cfg.q0 + i[0] - o[0],
        q >= 0,
    ]
    if n > 1:
        constraints.append(q[1:] == q[:-1] + i[1:] - o[1:])
    if cfg.nonnegative_flows:
        constraints.extend([i >= 0, o >= 0])

    problem = cp.Problem(cp.Minimize(sum(objective_terms)), constraints)
    solve_kwargs = {
        "solver": cfg.solver,
    }
    if cfg.solver.upper() == "OSQP":
        solve_kwargs.update(
            {
                "eps_abs": cfg.eps_abs,
                "eps_rel": cfg.eps_rel,
                "max_iter": cfg.max_iter,
            }
        )
    problem.solve(**solve_kwargs)

    if problem.status not in {"optimal", "optimal_inaccurate"}:
        raise RuntimeError(f"QP solve failed with status: {problem.status}")

    i_val = np.asarray(i.value).reshape(-1)
    o_val = np.asarray(o.value).reshape(-1)
    q_val = np.asarray(q.value).reshape(-1)
    if cfg.nonnegative_flows:
        i_val = np.maximum(i_val, 0.0)
        o_val = np.maximum(o_val, 0.0)
    q_val = np.maximum(q_val, 0.0)

    out = pd.DataFrame(
        {
            "minute_start_utc": work["minute_start_utc"],
            "in_count_measured": in_measured,
            "out_count_measured": out_measured,
            "in_count_corrected": i_val,
            "out_count_corrected": o_val,
            "occupancy_corrected_end": q_val,
        }
    )
    return out

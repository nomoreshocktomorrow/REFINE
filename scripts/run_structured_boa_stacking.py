import argparse
import pickle
import pathlib
import sys

import numpy as np
import pandas as pd
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import preprocess_backtest_data, get_yearmons
from scripts.run_walkforward_stacking import (
    load_pickle,
    monthly_return_from_backtest,
    average_weight_frame,
)


DATA_UNIVERSE = {
    "universe1": "universe1",
    "universe2": "universe2",
    "universe3": "universe3",
    "universe4": "universe4",
    "nasdaq100": "nasdaq100",
}

REFERENCE_UNIVERSE = {
    "universe1": "universe1",
    "universe2": "universe2",
    "universe3": "universe3",
    "universe4": "universe4",
    "nasdaq100": "nasdaq100",
}


def resolve_path(path_value: str) -> pathlib.Path:
    path = pathlib.Path(path_value)
    if path.is_absolute():
        return path
    return ROOT / path


def parse_market_data_map(value: str) -> dict[str, str]:
    out = {}
    for item in str(value or "").split(","):
        item = item.strip()
        if not item or "=" not in item:
            continue
        key, val = item.split("=", 1)
        out[key.strip()] = val.strip()
    return out


def data_universe_for(universe: str, args: argparse.Namespace) -> str:
    custom = parse_market_data_map(getattr(args, "market_data_map", ""))
    return custom.get(universe, DATA_UNIVERSE[universe])


def build_input_panel(universe: str, runs: list[dict]) -> tuple[list[str], pd.DataFrame, dict, pd.DataFrame, list[str]]:
    base_labels = []
    returns = []
    weight_frames = {}
    metadata_rows = []
    for rec in runs:
        label = rec["label"]
        base_labels.append(label)
        backtest = load_pickle(rec["backtest"])
        weights = load_pickle(rec["weights"])
        returns.append(monthly_return_from_backtest(backtest).rename(label))
        weight_frames[label] = average_weight_frame(weights)
        metadata_rows.append({
            "expert": label,
            "source": "base",
            "objective": rec["objective"],
            "model": rec["model"],
            "members": label,
        })
    ret_df = pd.concat(returns, axis=1).sort_index().loc[:, base_labels]
    tickers = list(next(iter(weight_frames.values())).columns)
    labels = list(base_labels)
    metadata = pd.DataFrame(metadata_rows)
    return labels, ret_df.loc[:, labels], weight_frames, metadata, tickers


def compose_final_weight(month: str, labels: list[str], coeffs: dict[str, float], weight_frames: dict, tickers: list[str]) -> np.ndarray:
    out = np.zeros(len(tickers), dtype=float)
    for label in labels:
        frame = weight_frames[label]
        if month in frame.index:
            vals = frame.loc[month].reindex(tickers).astype(float).to_numpy()
        else:
            vals = np.zeros(len(tickers), dtype=float)
        out += float(coeffs.get(label, 0.0)) * vals
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    out[out < 0.0] = 0.0
    if out.sum() > 1e-12:
        out = out / out.sum()
    return out


def safe_norm(weights: np.ndarray, valid: np.ndarray) -> np.ndarray:
    weights = np.where(valid, weights, 0.0)
    total = float(weights.sum())
    if total > 1e-12:
        return weights / total
    count = int(valid.sum())
    if count > 0:
        return np.where(valid, 1.0 / count, 0.0)
    return weights


def compute_summary_from_logs(logs: list[float], periods_per_year: float = 252.0) -> dict:
    logs_arr = np.asarray(logs, dtype=float)
    logs_arr = logs_arr[np.isfinite(logs_arr)]
    if logs_arr.size == 0:
        return {
            "total_return": np.nan,
            "annualized_return": np.nan,
            "annualized_vol": np.nan,
            "sharpe": np.nan,
            "sortino": np.nan,
            "max_drawdown": np.nan,
        }
    equity = np.cumprod(np.exp(logs_arr))
    downside = np.minimum(logs_arr, 0.0)
    return {
        "total_return": float(equity[-1]),
        "annualized_return": float(np.exp(logs_arr.mean() * float(periods_per_year))),
        "annualized_vol": float(logs_arr.std(ddof=0) * np.sqrt(float(periods_per_year))),
        "sharpe": float(logs_arr.mean() * np.sqrt(float(periods_per_year)) / (logs_arr.std(ddof=0) + 1e-12)),
        "sortino": float(logs_arr.mean() * np.sqrt(float(periods_per_year)) / (np.sqrt(np.mean(downside * downside)) + 1e-12)),
        "max_drawdown": float((equity / np.maximum.accumulate(equity) - 1.0).min()),
    }


def evaluate_weight_frame(data_universe: str, weight_df: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    _, data, missing_mask = preprocess_backtest_data(str(ROOT / "data" / f"{data_universe}.csv"), torch.device("cpu"))
    tickers = list(data["log_tr"].columns)
    weight_df = weight_df.reindex(columns=tickers).fillna(0.0)
    date2ind = {x: ind for ind, x in enumerate(data.index)}
    ym2idx = {ym: (date2ind[data.loc[ym].index[0]], date2ind[data.loc[ym].index[-1]]) for ym in get_yearmons(data)}
    months = [m for m in get_yearmons(data) if args.test_start <= m <= args.test_end]
    daily_logs = []
    rows = []
    equity = 1.0
    prev_w = None
    for month in months:
        if month not in weight_df.index:
            continue
        start_idx, end_idx = ym2idx[month]
        t0 = start_idx + int(args.lag)
        t1 = min(end_idx + int(args.lag), len(data.index) - 1)
        if t0 > t1:
            continue
        seg = data["log_tr"].iloc[t0:t1 + 1].to_numpy(dtype=float)
        mask = missing_mask[month].cpu().numpy().astype(bool) if month in missing_mask else np.zeros(len(tickers), dtype=bool)
        cur_ret = np.exp(np.nan_to_num(np.where(mask.reshape(1, -1), 0.0, seg), nan=0.0, posinf=0.0, neginf=0.0))
        valid = ~mask
        w = np.nan_to_num(weight_df.loc[month].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        w[w < 0.0] = 0.0
        if float(np.sum(np.where(valid, w, 0.0))) <= 1e-12:
            month_log = 0.0
            if cur_ret.shape[0] > 0:
                daily_logs.extend([0.0] * int(cur_ret.shape[0]))
            rows.append({"month": month, "log_return": month_log, "simple_return": 0.0, "equity": equity})
            continue
        w = safe_norm(w, valid)
        if cur_ret.shape[0] > 0:
            if prev_w is None:
                cur_ret[0, :] = cur_ret[0, :] - w * float(args.cost)
            else:
                cur_ret[0, :] = cur_ret[0, :] - np.abs(w - prev_w) * float(args.cost)
        mix = np.clip(np.nan_to_num((cur_ret * w.reshape(1, -1)).sum(axis=1), nan=1.0, posinf=1e12, neginf=1e-12), 1e-12, None)
        day_logs = np.log(mix)
        month_log = float(day_logs.sum())
        equity *= float(np.exp(month_log))
        daily_logs.extend(day_logs.tolist())
        rows.append({"month": month, "log_return": month_log, "simple_return": float(np.exp(month_log) - 1.0), "equity": equity})
        prev_w = safe_norm(w * cur_ret.prod(axis=0), valid)
    return pd.DataFrame(rows), compute_summary_from_logs(daily_logs, periods_per_year=252.0)


def run_equal_baseline(universe: str, base_labels: list[str], expert_returns: pd.DataFrame, weight_frames: dict, tickers: list[str], args: argparse.Namespace, out_dir: pathlib.Path, method: str = "equal12") -> dict:
    months = [m for m in expert_returns.index if args.test_start <= m <= args.test_end]
    rows = []
    coeff_rows = []
    n = len(base_labels)
    for month in months:
        coeffs = {label: 1.0 / n for label in base_labels}
        rows.append(pd.Series(compose_final_weight(month, base_labels, coeffs, weight_frames, tickers), index=tickers, name=month))
        coeff_rows.append({"month": month, **coeffs})
    asset_weight_df = pd.DataFrame(rows)
    monthly_eval, summary = evaluate_weight_frame(data_universe_for(universe, args), asset_weight_df, args)
    prefix = f"{REFERENCE_UNIVERSE[universe]}_{method}_{args.test_start.replace('-', '')}_{args.test_end.replace('-', '')}"
    asset_weight_df.to_csv(out_dir / f"{prefix}_asset_weights.csv")
    pd.DataFrame(coeff_rows).to_csv(out_dir / f"{prefix}_expert_coefficients.csv", index=False)
    monthly_eval.to_csv(out_dir / f"{prefix}_monthly_eval.csv", index=False)
    with open(out_dir / f"{prefix}_all_weights.pkl", "wb") as f:
        pickle.dump([asset_weight_df], f)
    item = {
        "universe": REFERENCE_UNIVERSE[universe],
        "method": method,
        "n_experts": len(base_labels),
        "n_base_experts": len(base_labels),
        "n_families": np.nan,
        "warmup_months": np.nan,
        "test_months": len(months),
        "eta_family_used": np.nan,
        "eta_expert_used": np.nan,
        "mean_hhi": 1.0 / n,
        "weights_pkl": str(out_dir / f"{prefix}.pkl"),
        "prefix": prefix,
    }
    item.update(summary)
    pd.DataFrame([item]).to_csv(out_dir / f"{prefix}_summary.csv", index=False)
    return item

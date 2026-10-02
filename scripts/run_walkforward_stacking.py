import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


OBJECTIVES = ["maxlogutility", "maxsharpe", "maxsortino", "mincvar"]
MODELS = ["lstm", "mamba", "trf"]
DEFAULT_UNIVERSES = ["universe1", "universe2", "universe3", "universe4", "nasdaq100"]


def parse_file_name(path: Path) -> tuple[str, str, str, str]:
    name = path.name
    if name.endswith("_all_weights.pkl"):
        kind = "weights"
    elif name.endswith("_all_backtest.pkl"):
        kind = "backtest"
    elif name.endswith("_all_logits.pkl"):
        kind = "logits"
    else:
        kind = "other"
    objective = next((x for x in OBJECTIVES if x in name), "unknown")
    model = next((x for x in MODELS if f"_{x}_" in name), "unknown")
    if name.startswith("nasdaq100"):
        universe = "nasdaq100"
    else:
        universe = next((x for x in DEFAULT_UNIVERSES if name.startswith(x)), "unknown")
    return universe, objective, model, kind


def discover_runs(data_dir: Path) -> dict[str, list[dict]]:
    grouped = defaultdict(dict)
    for path in sorted(data_dir.glob("*.pkl")):
        universe, objective, model, kind = parse_file_name(path)
        if universe == "unknown" or objective == "unknown" or model == "unknown" or kind == "other":
            continue
        stem = path.name
        for suffix in ["_all_weights.pkl", "_all_backtest.pkl", "_all_logits.pkl"]:
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        grouped[(universe, stem)].setdefault("universe", universe)
        grouped[(universe, stem)].setdefault("objective", objective)
        grouped[(universe, stem)].setdefault("model", model)
        grouped[(universe, stem)][kind] = path
    runs_by_universe = defaultdict(list)
    label_counts = Counter()
    for (universe, stem), rec in sorted(grouped.items()):
        if "weights" not in rec or "backtest" not in rec:
            continue
        base_label = f"{rec['objective']}_{rec['model']}"
        label_counts[(universe, base_label)] += 1
        suffix = label_counts[(universe, base_label)]
        label = base_label if suffix == 1 else f"{base_label}_v{suffix}"
        rec["label"] = label
        rec["stem"] = stem
        runs_by_universe[universe].append(rec)
    return dict(runs_by_universe)


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def monthly_return_from_backtest(backtest: pd.DataFrame) -> pd.Series:
    df = backtest.copy().astype(float)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    gross = (1.0 + df).clip(lower=1e-12)
    monthly_seed = gross.groupby(gross.index.strftime("%Y-%m")).prod() - 1.0
    return monthly_seed.mean(axis=1)


def average_weight_frame(weight_obj) -> pd.DataFrame:
    frames = weight_obj if isinstance(weight_obj, list) else [weight_obj]
    if len(frames) == 0:
        raise ValueError("empty weight object")
    base_index = frames[0].index
    base_columns = frames[0].columns
    values = []
    for df in frames:
        values.append(df.reindex(index=base_index, columns=base_columns).astype(float).values)
    arr = np.nanmean(np.stack(values, axis=0), axis=0)
    out = pd.DataFrame(arr, index=base_index, columns=base_columns)
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out[out < 0.0] = 0.0
    sums = out.sum(axis=1).replace(0.0, np.nan)
    out = out.div(sums, axis=0).fillna(0.0)
    return out

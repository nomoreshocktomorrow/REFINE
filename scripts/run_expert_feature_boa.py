import argparse
import json
import pathlib
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_walkforward_stacking import DEFAULT_UNIVERSES, discover_runs
from scripts.run_boa_stacking import auto_eta, boa_prob, log_loss_from_return, stable_softmax
from scripts.run_structured_boa_stacking import (
    REFERENCE_UNIVERSE,
    build_input_panel,
    compose_final_weight,
    data_universe_for,
    evaluate_weight_frame,
    resolve_path,
    run_equal_baseline,
)

FINAL_PRESET_NAME = "final_202505_revcmp3"

FINAL_PRESET_CONFIGS = {
    "universe1": {
        "recipe": {"revcmp3": 1.0},
        "eta_expert": 0.10,
        "forget": 0.55,
        "feature_prior_mix": 0.30,
        "feature_scale": 0.75,
        "data_dir": "",
        "market": "universe1",
    },
    "universe2": {
        "recipe": {"revcmp3": 1.0},
        "eta_expert": 0.10,
        "forget": 0.65,
        "feature_prior_mix": 0.50,
        "feature_scale": 1.25,
        "data_dir": "",
        "market": "universe2",
    },
    "universe3": {
        "recipe": {"revcmp3": 1.0},
        "eta_expert": 0.10,
        "forget": 0.55,
        "feature_prior_mix": 0.60,
        "feature_scale": 1.50,
        "data_dir": "",
        "market": "universe3",
    },
    "universe4": {
        "recipe": {"revcmp3": 1.0},
        "eta_expert": 0.10,
        "forget": 0.60,
        "feature_prior_mix": 0.60,
        "feature_scale": 1.25,
        "data_dir": "",
        "market": "universe4",
    },
    "nasdaq100": {
        "recipe": {"revcmp3": 1.0},
        "eta_expert": 0.10,
        "forget": 0.85,
        "feature_prior_mix": 0.60,
        "feature_scale": 1.50,
        "data_dir": "",
        "market": "nasdaq Top100",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--market-data-map", default="")
    parser.add_argument("--universes", default="all")
    parser.add_argument("--test-start", default="2019-11")
    parser.add_argument("--test-end", default="2025-05")
    parser.add_argument("--lag", type=int, default=2)
    parser.add_argument("--cost", type=float, default=0.004)
    parser.add_argument("--eta-expert", type=float, default=0.0)
    parser.add_argument("--second-order-scale", type=float, default=0.05)
    parser.add_argument("--forget", type=float, default=0.80)
    parser.add_argument("--max-weight", type=float, default=0.70)
    parser.add_argument("--downside-penalty", type=float, default=0.1)
    parser.add_argument("--feature-prior-mix", type=float, default=0.50)
    parser.add_argument("--feature-scale", type=float, default=1.0)
    return parser.parse_args()


def parse_universes(value: str) -> list[str]:
    if str(value).strip().lower() == "all":
        return list(DEFAULT_UNIVERSES)
    return [x.strip() for x in str(value).split(",") if x.strip()]


def preset_config_for(universe: str) -> dict:
    return dict(FINAL_PRESET_CONFIGS.get(universe, {}))


def recipe_for(universe: str) -> dict[str, float]:
    return dict(preset_config_for(universe).get("recipe", {}))


def args_for_universe(universe: str, args: argparse.Namespace) -> argparse.Namespace:
    local_args = argparse.Namespace(**vars(args))
    preset_config = preset_config_for(universe)
    if preset_config:
        local_args.eta_expert = float(preset_config["eta_expert"])
        local_args.forget = float(preset_config["forget"])
        local_args.feature_prior_mix = float(preset_config["feature_prior_mix"])
        local_args.feature_scale = float(preset_config["feature_scale"])
        if preset_config.get("market") and not str(args.market_data_map).strip():
            local_args.market_data_map = f"{universe}={preset_config['market']}"
    return local_args


def data_dir_for_universe(universe: str, args: argparse.Namespace) -> pathlib.Path:
    if str(args.data_dir).strip():
        return resolve_path(args.data_dir)
    preset_config = preset_config_for(universe)
    if preset_config.get("data_dir"):
        return resolve_path(preset_config["data_dir"])
    return resolve_path("")


def config_row_for_universe(universe: str, data_dir: pathlib.Path, recipe: dict[str, float], args: argparse.Namespace) -> dict:
    return {
        "universe": universe,
        "preset_used": FINAL_PRESET_NAME,
        "data_dir": str(data_dir),
        "market_data_map": str(args.market_data_map),
        "feature_recipe": format_recipe(recipe),
        "eta_expert": float(args.eta_expert),
        "forget": float(args.forget),
        "feature_prior_mix": float(args.feature_prior_mix),
        "feature_scale": float(args.feature_scale),
        "second_order_scale": float(args.second_order_scale),
        "max_weight": float(args.max_weight),
        "downside_penalty": float(args.downside_penalty),
        "lag": int(args.lag),
        "cost": float(args.cost),
        "test_start": str(args.test_start),
        "test_end": str(args.test_end),
    }


def output_row_from_summary(item: dict) -> dict:
    return {
        "universe": item.get("universe"),
        "method": item.get("method"),
        "prefix": item.get("prefix"),
        "weights_pkl": item.get("weights_pkl"),
    }


def format_recipe(recipe: dict[str, float]) -> str:
    if not recipe:
        return "uniform"
    return ",".join(f"{name}:{float(weight):g}" for name, weight in recipe.items())


def cross_sectional_zscore(values: np.ndarray) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    mask = np.isfinite(vals)
    if mask.sum() <= 1:
        return np.zeros_like(vals, dtype=float)
    mean = float(np.nanmean(vals[mask]))
    std = float(np.nanstd(vals[mask]))
    if std <= 1e-12 or not np.isfinite(std):
        return np.zeros_like(vals, dtype=float)
    out = np.zeros_like(vals, dtype=float)
    out[mask] = (vals[mask] - mean) / std
    return out


def feature_scores(hist: pd.DataFrame, name: str) -> np.ndarray:
    if hist.empty:
        return np.zeros(hist.shape[1], dtype=float)
    if name == "revcmp3":
        h = hist.tail(3)
        return -((1.0 + h).prod(axis=0) - 1.0).to_numpy(dtype=float)
    raise ValueError(f"Unknown feature: {name}")


def build_feature_prior(hist: pd.DataFrame, labels: list[str], recipe: dict[str, float], feature_scale: float, feature_prior_mix: float) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    n = len(labels)
    uniform = np.full(n, 1.0 / n, dtype=float)
    if not recipe or hist.empty:
        return uniform, {}, np.zeros(n, dtype=float)
    combined = np.zeros(n, dtype=float)
    components = {}
    used = False
    for name, weight in recipe.items():
        score = feature_scores(hist, name)
        zscore = cross_sectional_zscore(score)
        components[name] = zscore
        if np.any(np.isfinite(zscore)) and abs(float(weight)) > 0.0:
            combined += float(weight) * zscore
            used = True
    if not used:
        return uniform, components, combined
    raw = stable_softmax(float(feature_scale) * combined)
    mix = float(np.clip(feature_prior_mix, 0.0, 1.0))
    prior = (1.0 - mix) * uniform + mix * raw
    prior = np.asarray(prior, dtype=float)
    prior = prior / max(float(prior.sum()), 1e-12)
    return prior, components, combined


def apply_coeff_blend(coeffs: dict[str, float], labels: list[str], args: argparse.Namespace) -> dict[str, float]:
    out = {label: float(coeffs.get(label, 0.0)) for label in labels}
    total = sum(out.values())
    if total > 1e-12:
        out = {k: v / total for k, v in out.items()}
    return out


def aggregate_with_valid(weights: np.ndarray, values: np.ndarray) -> tuple[float, np.ndarray]:
    valid = np.isfinite(values)
    if not valid.any():
        return 0.0, np.zeros_like(weights, dtype=float)
    w = np.asarray(weights, dtype=float)[valid]
    total = float(w.sum())
    if total <= 1e-12 or not np.isfinite(total):
        w = np.full(valid.sum(), 1.0 / valid.sum(), dtype=float)
    else:
        w = w / total
    return float(np.dot(w, values[valid])), w


def update_expert_state(month: str, labels: list[str], expert_returns: pd.DataFrame, probs: np.ndarray, regret: np.ndarray, variation: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    row = expert_returns.loc[month, labels].astype(float).to_numpy(dtype=float)
    agg_ret, _ = aggregate_with_valid(probs, row)
    agg_loss = log_loss_from_return(agg_ret, args.downside_penalty)
    losses = np.full(len(labels), np.nan, dtype=float)
    valid = np.isfinite(row)
    losses[valid] = np.array([log_loss_from_return(x, args.downside_penalty) for x in row[valid]], dtype=float)
    deltas = np.zeros(len(labels), dtype=float)
    deltas[valid] = agg_loss - losses[valid]
    new_regret = float(args.forget) * regret
    new_variation = float(args.forget) * variation
    new_regret[valid] += deltas[valid]
    new_variation[valid] += deltas[valid] * deltas[valid]
    return new_regret, new_variation


def select_probabilities(labels: list[str], prior: np.ndarray, regret: np.ndarray, variation: np.ndarray, eta_expert: float, args: argparse.Namespace) -> tuple[np.ndarray, dict[str, float]]:
    probs = boa_prob(prior, regret, variation, eta_expert, args.second_order_scale, args.max_weight)
    coeffs = {label: float(probs[i]) for i, label in enumerate(labels)}
    coeffs = apply_coeff_blend(coeffs, labels, args)
    probs = np.array([coeffs[label] for label in labels], dtype=float)
    probs = probs / max(float(probs.sum()), 1e-12)
    return probs, coeffs


def run_expert_feature_boa(universe: str, base_labels: list[str], expert_returns: pd.DataFrame, weight_frames: dict, tickers: list[str], recipe: dict[str, float], args: argparse.Namespace, out_dir: pathlib.Path, method: str = "expert_feature_boa") -> dict:
    all_months = [m for m in expert_returns.index if m <= args.test_end]
    warmup_months = [m for m in all_months if m < args.test_start]
    test_months = [m for m in all_months if args.test_start <= m <= args.test_end]
    horizon = len(all_months)
    eta_expert = float(args.eta_expert) if float(args.eta_expert) > 0.0 else auto_eta(len(base_labels), horizon)
    regret = np.zeros(len(base_labels), dtype=float)
    variation = np.zeros(len(base_labels), dtype=float)

    for month in warmup_months:
        pos = all_months.index(month)
        hist = expert_returns.iloc[:pos].loc[:, base_labels]
        prior, _, _ = build_feature_prior(hist, base_labels, recipe, args.feature_scale, args.feature_prior_mix)
        probs, _ = select_probabilities(base_labels, prior, regret, variation, eta_expert, args)
        regret, variation = update_expert_state(month, base_labels, expert_returns, probs, regret, variation, args)

    weight_rows = []
    coeff_rows = []
    prior_rows = []
    score_rows = []
    hhi_list = []
    for month in test_months:
        pos = all_months.index(month)
        hist = expert_returns.iloc[:pos].loc[:, base_labels]
        prior, _, combined = build_feature_prior(hist, base_labels, recipe, args.feature_scale, args.feature_prior_mix)
        probs, coeffs = select_probabilities(base_labels, prior, regret, variation, eta_expert, args)
        final_weight = compose_final_weight(month, base_labels, coeffs, weight_frames, tickers)
        weight_rows.append(pd.Series(final_weight, index=tickers, name=month))
        coeff_rows.append({"month": month, **coeffs})
        prior_rows.append({"month": month, **{label: float(prior[i]) for i, label in enumerate(base_labels)}})
        score_rows.append({"month": month, **{label: float(combined[i]) for i, label in enumerate(base_labels)}})
        hhi_list.append(float(np.dot(probs, probs)))
        regret, variation = update_expert_state(month, base_labels, expert_returns, probs, regret, variation, args)

    asset_weight_df = pd.DataFrame(weight_rows)
    monthly_eval, summary = evaluate_weight_frame(data_universe_for(universe, args), asset_weight_df, args)
    prefix = f"{REFERENCE_UNIVERSE[universe]}_{method}_{args.test_start.replace('-', '')}_{args.test_end.replace('-', '')}"
    asset_weight_df.to_csv(out_dir / f"{prefix}_asset_weights.csv")
    pd.DataFrame(coeff_rows).to_csv(out_dir / f"{prefix}_expert_coefficients.csv", index=False)
    pd.DataFrame(prior_rows).to_csv(out_dir / f"{prefix}_feature_prior.csv", index=False)
    pd.DataFrame(score_rows).to_csv(out_dir / f"{prefix}_feature_scores.csv", index=False)
    monthly_eval.to_csv(out_dir / f"{prefix}_monthly_eval.csv", index=False)
    with open(out_dir / f"{prefix}_all_weights.pkl", "wb") as f:
        pickle.dump([asset_weight_df], f)
    item = {
        "universe": REFERENCE_UNIVERSE[universe],
        "method": method,
        "n_experts": len(base_labels),
        "n_base_experts": len(base_labels),
        "n_families": 1,
        "warmup_months": len(warmup_months),
        "test_months": len(test_months),
        "eta_family_used": np.nan,
        "eta_expert_used": eta_expert,
        "mean_hhi": float(np.mean(hhi_list)) if hhi_list else np.nan,
        "feature_recipe": format_recipe(recipe),
        "feature_prior_mix": float(args.feature_prior_mix),
        "feature_scale": float(args.feature_scale),
        "weights_pkl": str(out_dir / f"{prefix}_all_weights.pkl"),
        "prefix": prefix,
    }
    item.update(summary)
    pd.DataFrame([item]).to_csv(out_dir / f"{prefix}_summary.csv", index=False)
    return item


def main() -> None:
    args = parse_args()
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    config_rows = []
    output_rows = []
    runs_cache = {}
    for universe in parse_universes(args.universes):
        local_args = args_for_universe(universe, args)
        data_dir = data_dir_for_universe(universe, local_args)
        cache_key = str(data_dir)
        if cache_key not in runs_cache:
            runs_cache[cache_key] = discover_runs(data_dir)
        runs_by_universe = runs_cache[cache_key]
        runs = runs_by_universe.get(universe, [])
        if len(runs) == 0:
            print(f"Skipping {universe}: no complete base experts in {data_dir}")
            continue
        labels, expert_returns, weight_frames, metadata, tickers = build_input_panel(universe, runs)
        base_labels = metadata.loc[metadata["source"].eq("base"), "expert"].tolist()
        expert_returns = expert_returns.loc[:, base_labels]
        recipe = recipe_for(universe)
        config_rows.append(config_row_for_universe(universe, data_dir, recipe, local_args))
        print(
            f"Running {universe}: {len(base_labels)} base experts, recipe={format_recipe(recipe)}, "
            f"eta={float(local_args.eta_expert):g}, forget={float(local_args.forget):g}, data_dir={data_dir}"
        )
        equal_item = run_equal_baseline(universe, base_labels, expert_returns, weight_frames, tickers, local_args, out_dir)
        equal_item.update({"preset_used": FINAL_PRESET_NAME})
        summaries.append(equal_item)
        output_rows.append(output_row_from_summary(equal_item))
        boa_item = run_expert_feature_boa(universe, base_labels, expert_returns, weight_frames, tickers, recipe, local_args, out_dir)
        boa_item.update({"preset_used": FINAL_PRESET_NAME})
        summaries.append(boa_item)
        output_rows.append(output_row_from_summary(boa_item))
    if summaries:
        summary_df = pd.DataFrame(summaries)
        summary_path = out_dir / f"all_expert_feature_boa_{args.test_start.replace('-', '')}_{args.test_end.replace('-', '')}_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        config_df = pd.DataFrame(config_rows)
        config_path = out_dir / f"all_expert_feature_boa_{args.test_start.replace('-', '')}_{args.test_end.replace('-', '')}_run_config.csv"
        config_df.to_csv(config_path, index=False)
        outputs_path = out_dir / f"all_expert_feature_boa_{args.test_start.replace('-', '')}_{args.test_end.replace('-', '')}_output_files.csv"
        pd.DataFrame(output_rows).to_csv(outputs_path, index=False)
        manifest_path = out_dir / f"all_expert_feature_boa_{args.test_start.replace('-', '')}_{args.test_end.replace('-', '')}_run_config.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump({"global_args": vars(args), "per_universe": config_rows}, f, ensure_ascii=False, indent=2)
        print(summary_df)
        print(f"Saved summary to {summary_path}")
        print(f"Saved run config to {config_path}")
        print(f"Saved output file manifest to {outputs_path}")


if __name__ == "__main__":
    main()

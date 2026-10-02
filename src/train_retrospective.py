import tqdm
import random
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

def _compute_marginal_downside_risk_score(
    monthly_asset_simple_returns: Dict[str, np.ndarray],
    yearmons: List[str],
    current_idx: int,
    valid_mask: np.ndarray,
    weight_mean: np.ndarray,
    downside_lookback_months: int,
    downside_target: float,
) -> np.ndarray:
    num_item = weight_mean.shape[0]
    downside_score = np.zeros(num_item, dtype=np.float32)

    history = []
    start_idx = max(0, current_idx - downside_lookback_months)
    for hist_idx in range(start_idx, current_idx):
        hist_yearmon = yearmons[hist_idx]
        if hist_yearmon in monthly_asset_simple_returns:
            history.append(monthly_asset_simple_returns[hist_yearmon])

    if len(history) < 2 or valid_mask.sum() < 2:
        return downside_score

    hist_returns = np.stack(history, axis=0)[:, valid_mask].astype(np.float32)
    if hist_returns.shape[0] < 2 or hist_returns.shape[1] < 2:
        return downside_score

    downside_returns = np.minimum(hist_returns - downside_target, 0.0).astype(np.float32)
    if not np.any(np.abs(downside_returns) > 1e-12):
        return downside_score

    semi_cov = (downside_returns.T @ downside_returns) / max(downside_returns.shape[0] - 1, 1)
    semi_cov = semi_cov.astype(np.float32)
    semi_cov = semi_cov + np.eye(semi_cov.shape[0], dtype=np.float32) * 1e-6
    marginal_downside = semi_cov @ weight_mean[valid_mask].astype(np.float32)
    if not np.any(np.abs(marginal_downside) > 1e-12):
        return downside_score

    valid_indices = np.where(valid_mask)[0]
    low_to_high_downside = valid_indices[np.argsort(marginal_downside)]
    downside_score[low_to_high_downside] = np.linspace(1.0, -1.0, valid_mask.sum(), dtype=np.float32)
    return downside_score


def _compute_log_utility_score(
    month_simple_returns: np.ndarray,
    valid_mask: np.ndarray,
    utility_floor: float,
    utility_cap: float,
    utility_score_clip: float,
) -> np.ndarray:
    num_item = month_simple_returns.shape[0]
    utility_score = np.zeros(num_item, dtype=np.float32)

    if valid_mask.sum() < 2:
        return utility_score

    valid_returns = np.clip(
        month_simple_returns[valid_mask].astype(np.float32),
        utility_floor,
        utility_cap,
    )
    valid_utility = np.log1p(valid_returns).astype(np.float32)
    valid_utility = np.nan_to_num(valid_utility, nan=0.0, posinf=0.0, neginf=0.0)
    utility_std = float(valid_utility.std())
    if utility_std < 1e-8:
        return utility_score

    valid_utility = (valid_utility - valid_utility.mean()) / (utility_std + 1e-8)
    valid_utility = np.clip(valid_utility, -utility_score_clip, utility_score_clip)
    utility_score[np.where(valid_mask)[0]] = valid_utility.astype(np.float32)
    return utility_score


def _compute_log_utility_bad_month_weight(
    month_simple_returns: np.ndarray,
    valid_mask: np.ndarray,
    weight_mean: np.ndarray,
    utility_floor: float,
    utility_cap: float,
    utility_bad_month_weight: float,
    utility_month_weight_cap: float,
) -> float:
    if utility_bad_month_weight <= 0:
        return 1.0

    valid_weight = np.where(valid_mask, weight_mean.astype(np.float32), 0.0)
    weight_sum = float(valid_weight.sum())
    if weight_sum <= 1e-12:
        return 1.0
    valid_weight = valid_weight / weight_sum

    portfolio_simple_return = float((valid_weight[valid_mask] * month_simple_returns[valid_mask]).sum())
    portfolio_simple_return = float(np.clip(portfolio_simple_return, utility_floor, utility_cap))
    portfolio_utility = float(np.log1p(portfolio_simple_return))
    bad_month_penalty = max(-portfolio_utility, 0.0)
    month_weight = 1.0 + utility_bad_month_weight * bad_month_penalty
    month_weight_cap = max(float(utility_month_weight_cap), 1.0)
    return float(min(month_weight, month_weight_cap))


def _normalize_reference_weight(
    reference_weight: Optional[np.ndarray],
    valid_mask: np.ndarray,
    fallback_weight: np.ndarray,
) -> np.ndarray:
    if reference_weight is None:
        return fallback_weight.astype(np.float32)

    normalized_reference = np.where(valid_mask, reference_weight.astype(np.float32), 0.0)
    reference_sum = float(normalized_reference.sum())
    if reference_sum <= 1e-12:
        return fallback_weight.astype(np.float32)
    normalized_reference = normalized_reference / reference_sum
    return normalized_reference.astype(np.float32)


def _stack_cvar_daily_history(
    daily_asset_simple_returns: Dict[str, np.ndarray],
    yearmons: List[str],
    current_idx: int,
    cvar_lookback_months: int,
) -> Optional[np.ndarray]:
    history = []
    lookback = max(int(cvar_lookback_months), 1)
    start_idx = max(0, current_idx - lookback + 1)
    for hist_idx in range(start_idx, current_idx + 1):
        hist_yearmon = yearmons[hist_idx]
        hist_daily_returns = daily_asset_simple_returns.get(hist_yearmon)
        if hist_daily_returns is None or hist_daily_returns.ndim != 2 or hist_daily_returns.shape[0] == 0:
            continue
        history.append(hist_daily_returns.astype(np.float32))

    if len(history) == 0:
        return None

    return np.concatenate(history, axis=0)


def _compute_cvar_tail_score(
    daily_asset_simple_returns: Dict[str, np.ndarray],
    yearmons: List[str],
    current_idx: int,
    valid_mask: np.ndarray,
    weight_mean: np.ndarray,
    cvar_alpha: float,
    cvar_lookback_months: int,
    cvar_score_clip: float,
) -> np.ndarray:
    num_item = weight_mean.shape[0]
    cvar_score = np.zeros(num_item, dtype=np.float32)

    if valid_mask.sum() < 2:
        return cvar_score

    history_daily_returns = _stack_cvar_daily_history(
        daily_asset_simple_returns=daily_asset_simple_returns,
        yearmons=yearmons,
        current_idx=current_idx,
        cvar_lookback_months=cvar_lookback_months,
    )
    if history_daily_returns is None or history_daily_returns.ndim != 2 or history_daily_returns.shape[0] == 0:
        return cvar_score

    valid_weight = np.where(valid_mask, weight_mean.astype(np.float32), 0.0)
    weight_sum = float(valid_weight.sum())
    if weight_sum <= 1e-12:
        return cvar_score
    valid_weight = valid_weight[valid_mask] / weight_sum

    valid_daily_returns = history_daily_returns[:, valid_mask].astype(np.float32)
    portfolio_daily_returns = (valid_daily_returns * valid_weight.reshape(1, -1)).sum(axis=1)

    tail_fraction = float(np.clip(cvar_alpha, 1e-4, 0.5))
    tail_count = max(1, int(np.ceil(portfolio_daily_returns.shape[0] * tail_fraction)))
    tail_indices = np.argsort(portfolio_daily_returns)[:tail_count]
    tail_portfolio_returns = portfolio_daily_returns[tail_indices].astype(np.float32)

    severity = np.maximum(-tail_portfolio_returns, 0.0).astype(np.float32)
    if not np.any(severity > 1e-12):
        severity = np.ones_like(tail_portfolio_returns, dtype=np.float32)
    severity = severity / np.clip(severity.sum(), 1e-6, None)

    tail_asset_returns = valid_daily_returns[tail_indices]
    tail_relative_returns = tail_asset_returns - tail_portfolio_returns.reshape(-1, 1)
    valid_tail_score = (severity.reshape(-1, 1) * tail_relative_returns).sum(axis=0)

    tail_std = float(valid_tail_score.std())
    if tail_std < 1e-8:
        return cvar_score

    valid_tail_score = (valid_tail_score - valid_tail_score.mean()) / (tail_std + 1e-8)
    valid_tail_score = np.clip(valid_tail_score, -cvar_score_clip, cvar_score_clip)
    cvar_score[np.where(valid_mask)[0]] = valid_tail_score.astype(np.float32)
    return cvar_score


def _compute_cvar_bad_month_weight(
    daily_asset_simple_returns: Dict[str, np.ndarray],
    yearmons: List[str],
    current_idx: int,
    valid_mask: np.ndarray,
    weight_mean: np.ndarray,
    cvar_alpha: float,
    cvar_lookback_months: int,
    cvar_bad_month_weight: float,
    cvar_month_weight_cap: float,
) -> float:
    if cvar_bad_month_weight <= 0:
        return 1.0

    history_daily_returns = _stack_cvar_daily_history(
        daily_asset_simple_returns=daily_asset_simple_returns,
        yearmons=yearmons,
        current_idx=current_idx,
        cvar_lookback_months=cvar_lookback_months,
    )
    if history_daily_returns is None or history_daily_returns.ndim != 2 or history_daily_returns.shape[0] == 0:
        return 1.0

    valid_weight = np.where(valid_mask, weight_mean.astype(np.float32), 0.0)
    weight_sum = float(valid_weight.sum())
    if weight_sum <= 1e-12:
        return 1.0
    valid_weight = valid_weight[valid_mask] / weight_sum

    valid_daily_returns = history_daily_returns[:, valid_mask].astype(np.float32)
    portfolio_daily_returns = (valid_daily_returns * valid_weight.reshape(1, -1)).sum(axis=1)

    tail_fraction = float(np.clip(cvar_alpha, 1e-4, 0.5))
    tail_count = max(1, int(np.ceil(portfolio_daily_returns.shape[0] * tail_fraction)))
    tail_portfolio_returns = np.sort(portfolio_daily_returns)[:tail_count].astype(np.float32)
    cvar_loss = max(float(-tail_portfolio_returns.mean()), 0.0)

    month_weight = 1.0 + cvar_bad_month_weight * cvar_loss
    month_weight_cap = max(float(cvar_month_weight_cap), 1.0)
    return float(min(month_weight, month_weight_cap))


def _build_weighted_failure_signal(
    monthly_asset_simple_returns: Dict[str, np.ndarray],
    daily_asset_simple_returns: Optional[Dict[str, np.ndarray]],
    yearmons: List[str],
    current_idx: int,
    valid_mask: np.ndarray,
    weight_mean: np.ndarray,
    fair_share: np.ndarray,
    reference_weight: Optional[np.ndarray],
    month_return: np.ndarray,
    month_simple_return: np.ndarray,
    return_term_weight: float = 1.0,
    downside_term_weight: float = 0.0,
    downside_lookback_months: int = 12,
    downside_target: float = 0.0,
    utility_term_weight: float = 0.0,
    utility_floor: float = -0.95,
    utility_cap: float = 5.0,
    utility_score_clip: float = 3.0,
    utility_bad_month_weight: float = 2.0,
    utility_month_weight_cap: float = 3.0,
    cvar_term_weight: float = 0.0,
    cvar_alpha: float = 0.05,
    cvar_lookback_months: int = 1,
    cvar_score_clip: float = 3.0,
    cvar_bad_month_weight: float = 2.0,
    cvar_month_weight_cap: float = 3.0,
) -> np.ndarray:
    num_item = weight_mean.shape[0]
    valid_indices = np.where(valid_mask)[0]
    valid_return_rank = valid_indices[np.argsort(month_return[valid_mask])[::-1]]
    effective_reference_weight = _normalize_reference_weight(reference_weight, valid_mask, fair_share)
    overweight = weight_mean - effective_reference_weight

    return_score = np.zeros(num_item, dtype=np.float32)
    return_score[valid_return_rank] = np.linspace(1.0, -1.0, valid_mask.sum(), dtype=np.float32)

    return_failure_signal = np.where(
        overweight * return_score < 0,
        overweight * np.abs(return_score),
        0.0,
    )
    failure_signal = return_term_weight * return_failure_signal

    if downside_term_weight > 0:
        downside_score = _compute_marginal_downside_risk_score(
            monthly_asset_simple_returns=monthly_asset_simple_returns,
            yearmons=yearmons,
            current_idx=current_idx,
            valid_mask=valid_mask,
            weight_mean=weight_mean,
            downside_lookback_months=downside_lookback_months,
            downside_target=downside_target,
        )
        downside_failure_signal = np.where(
            overweight * downside_score < 0,
            overweight * np.abs(downside_score),
            0.0,
        )
        failure_signal = failure_signal + downside_term_weight * downside_failure_signal

    if utility_term_weight > 0:
        utility_score = _compute_log_utility_score(
            month_simple_returns=month_simple_return,
            valid_mask=valid_mask,
            utility_floor=utility_floor,
            utility_cap=utility_cap,
            utility_score_clip=utility_score_clip,
        )
        utility_failure_signal = np.where(
            overweight * utility_score < 0,
            overweight * np.abs(utility_score),
            0.0,
        )
        utility_failure_signal = utility_failure_signal * _compute_log_utility_bad_month_weight(
            month_simple_returns=month_simple_return,
            valid_mask=valid_mask,
            weight_mean=weight_mean,
            utility_floor=utility_floor,
            utility_cap=utility_cap,
            utility_bad_month_weight=utility_bad_month_weight,
            utility_month_weight_cap=utility_month_weight_cap,
        )
        failure_signal = failure_signal + utility_term_weight * utility_failure_signal

    if cvar_term_weight > 0 and daily_asset_simple_returns is not None:
        cvar_score = _compute_cvar_tail_score(
            daily_asset_simple_returns=daily_asset_simple_returns,
            yearmons=yearmons,
            current_idx=current_idx,
            valid_mask=valid_mask,
            weight_mean=weight_mean,
            cvar_alpha=cvar_alpha,
            cvar_lookback_months=cvar_lookback_months,
            cvar_score_clip=cvar_score_clip,
        )
        cvar_failure_signal = np.where(
            overweight * cvar_score < 0,
            overweight * np.abs(cvar_score),
            0.0,
        )
        cvar_failure_signal = cvar_failure_signal * _compute_cvar_bad_month_weight(
            daily_asset_simple_returns=daily_asset_simple_returns,
            yearmons=yearmons,
            current_idx=current_idx,
            valid_mask=valid_mask,
            weight_mean=weight_mean,
            cvar_alpha=cvar_alpha,
            cvar_lookback_months=cvar_lookback_months,
            cvar_bad_month_weight=cvar_bad_month_weight,
            cvar_month_weight_cap=cvar_month_weight_cap,
        )
        failure_signal = failure_signal + cvar_term_weight * cvar_failure_signal

    return failure_signal


def _diagnose_collective_failure_patterns(
    first_round_weights: List[pd.DataFrame],
    first_round_logits: List[pd.DataFrame],
    torch_target_data: torch.Tensor,
    yearmon2indices: Dict[str, Tuple[int, int]],
    missing_mask: Dict[str, torch.Tensor],
    yearmons: List[str],
    backtest_start: str,
    lagging: int,
    trading_cost: float,
    device: torch.device,
    target_weight: Dict[str, torch.Tensor],
    portfolio: str = 'maxsortino',
    consensus_threshold: float = 0.6,
    disagreement_floor: float = 0.1,
    return_term_weight: float = 1.0,
    downside_term_weight: float = 0.0,
    downside_lookback_months: int = 12,
    downside_target: float = 0.0,
    utility_term_weight: float = 0.0,
    utility_floor: float = -0.95,
    utility_cap: float = 5.0,
    utility_score_clip: float = 3.0,
    utility_bad_month_weight: float = 2.0,
    utility_month_weight_cap: float = 3.0,
    cvar_term_weight: float = 0.0,
    cvar_alpha: float = 0.05,
    cvar_lookback_months: int = 1,
    cvar_score_clip: float = 3.0,
    cvar_bad_month_weight: float = 2.0,
    cvar_month_weight_cap: float = 3.0,
) -> Dict[str, torch.Tensor]:
    """
    Diagnose collective failure patterns across seeds.
    
    Returns:
        failure_patterns: Dict[yearmon -> (num_item,)] tensor of correction signals
            - Positive: collective over-weight (should reduce)
            - Negative: collective under-weight (should increase)
            - Magnitude scaled by consensus strength and actual loss
    """
    num_repeat = len(first_round_weights)
    num_item = first_round_weights[0].shape[1]

    failure_patterns = {}
    monthly_asset_simple_returns = {}
    daily_asset_simple_returns = {}
    for yearmon in yearmons:
        if yearmon not in yearmon2indices:
            continue
        predict_start, predict_end = yearmon2indices[yearmon]
        month_daily_log_returns = torch_target_data[
            predict_start + lagging:predict_end + 1 + lagging
        ]
        month_log_return = month_daily_log_returns.sum(dim=0)
        monthly_asset_simple_returns[yearmon] = month_log_return.exp().cpu().numpy().astype(np.float32) - 1.0
        daily_asset_simple_returns[yearmon] = month_daily_log_returns.exp().cpu().numpy().astype(np.float32) - 1.0
    
    for idx, yearmon in enumerate(yearmons):
        if yearmon < backtest_start:
            continue
        
        if idx == 0:
            continue
        
        seed_weights = []
        seed_logits = []
        
        for rep in range(num_repeat):
            if yearmon in first_round_weights[rep].index:
                w = first_round_weights[rep].loc[yearmon].values
                l = first_round_logits[rep].loc[yearmon].values
                seed_weights.append(w)
                seed_logits.append(l)
        
        if len(seed_weights) < 2:
            continue
        
        seed_weights = np.stack(seed_weights, axis=0)
        seed_logits = np.stack(seed_logits, axis=0)
        
        mask = missing_mask[yearmon]
        valid_mask = (~mask.cpu().numpy()).astype(bool)
        if valid_mask.sum() == 0:
            continue

        seed_weights = np.where(valid_mask.reshape(1, -1), seed_weights, 0.0)
        seed_weight_sum = np.clip(seed_weights.sum(axis=1, keepdims=True), 1e-12, None)
        seed_weights = seed_weights / seed_weight_sum

        weight_std = seed_weights.std(axis=0)
        weight_mean = seed_weights.mean(axis=0)

        fair_share = np.zeros(num_item, dtype=np.float32)
        fair_share[valid_mask] = 1.0 / float(valid_mask.sum())
        reference_weight = None
        if portfolio == 'mincvar' and yearmon in target_weight:
            reference_weight = target_weight[yearmon].detach().cpu().numpy().astype(np.float32)
        normalized_reference_weight = _normalize_reference_weight(reference_weight, valid_mask, fair_share)
        weight_scale = np.maximum(weight_mean, normalized_reference_weight)

        disagreement = np.zeros(num_item, dtype=np.float32)
        disagreement[valid_mask] = weight_std[valid_mask] / (weight_scale[valid_mask] + 1e-6)
        disagreement = np.clip(disagreement, disagreement_floor, 10.0)

        predict_start, predict_end = yearmon2indices[yearmon]
        
        realized_returns = torch_target_data[predict_start + lagging:predict_end + 1 + lagging].exp()
        month_return = realized_returns.prod(dim=0).cpu().numpy()
        month_simple_return = monthly_asset_simple_returns[yearmon]
        overweight = weight_mean - normalized_reference_weight
        failure_signal = _build_weighted_failure_signal(
            monthly_asset_simple_returns=monthly_asset_simple_returns,
            daily_asset_simple_returns=daily_asset_simple_returns,
            yearmons=yearmons,
            current_idx=idx,
            valid_mask=valid_mask,
            weight_mean=weight_mean,
            fair_share=fair_share,
            reference_weight=normalized_reference_weight,
            month_return=month_return,
            month_simple_return=month_simple_return,
            return_term_weight=return_term_weight,
            downside_term_weight=downside_term_weight,
            downside_lookback_months=downside_lookback_months,
            downside_target=downside_target,
            utility_term_weight=utility_term_weight,
            utility_floor=utility_floor,
            utility_cap=utility_cap,
            utility_score_clip=utility_score_clip,
            utility_bad_month_weight=utility_bad_month_weight,
            utility_month_weight_cap=utility_month_weight_cap,
            cvar_term_weight=cvar_term_weight,
            cvar_alpha=cvar_alpha,
            cvar_lookback_months=cvar_lookback_months,
            cvar_score_clip=cvar_score_clip,
            cvar_bad_month_weight=cvar_bad_month_weight,
            cvar_month_weight_cap=cvar_month_weight_cap,
        )
        
        correction_strength = 1.0 / (1.0 + disagreement)
        
        collective_failure = failure_signal * correction_strength
        
        seed_overweight = seed_weights - normalized_reference_weight.reshape(1, -1)
        median_direction = np.sign(overweight)
        weight_direction_consensus = np.zeros(num_item, dtype=np.float32)
        active_direction = np.abs(median_direction) > 1e-12
        if np.any(active_direction):
            weight_direction_consensus[active_direction] = (
                np.sign(seed_overweight[:, active_direction]) == median_direction[active_direction]
            ).mean(axis=0)
        
        collective_failure = collective_failure * (weight_direction_consensus >= consensus_threshold)
        collective_failure = np.where(valid_mask, collective_failure, 0.0)
        active_failure = np.abs(collective_failure) > 1e-12
        if np.any(active_failure):
            collective_failure[active_failure] = collective_failure[active_failure] / np.clip(
                np.abs(collective_failure[active_failure]).mean(),
                1e-6,
                None,
            )

        failure_patterns[yearmon] = torch.tensor(
            collective_failure, dtype=torch.float32, device=device
        )
    
    return failure_patterns


def _get_causal_retrospective_pattern(
    failure_patterns: Dict[str, torch.Tensor],
    current_yearmon: str,
    current_missing_mask: torch.Tensor,
    lookback: int = 6,
    decay: float = 0.7,
) -> Optional[torch.Tensor]:
    past_yearmons = []
    for ym in sorted(failure_patterns.keys()):
        if ym >= current_yearmon:
            continue
        if torch.any(failure_patterns[ym].abs() > 1e-12):
            past_yearmons.append(ym)

    if len(past_yearmons) == 0:
        return None

    selected_yearmons = past_yearmons[-lookback:]
    patterns = []
    weights = []
    for offset, ym in enumerate(reversed(selected_yearmons)):
        patterns.append(failure_patterns[ym])
        weights.append(decay ** offset)

    weight_tensor = torch.tensor(weights, dtype=patterns[0].dtype, device=patterns[0].device)
    stacked_patterns = torch.stack(patterns, dim=0)
    aggregated_pattern = (stacked_patterns * weight_tensor.unsqueeze(-1)).sum(dim=0) / weight_tensor.sum().clamp_min(1e-6)
    aggregated_pattern = torch.where(current_missing_mask, torch.zeros_like(aggregated_pattern), aggregated_pattern)

    active_pattern = aggregated_pattern.abs() > 1e-12
    if not torch.any(active_pattern):
        return None

    aggregated_pattern = aggregated_pattern / aggregated_pattern[active_pattern].abs().mean().clamp_min(1e-6)
    return aggregated_pattern


def _retrospective_correction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    missing_mask: torch.Tensor,
    failure_pattern: Optional[torch.Tensor],
    kl_weight: float = 0.7,
    correction_weight: float = 0.3,
) -> torch.Tensor:
    """
    Combined loss: original loss + retrospective correction.
    
    Original loss (KL to teacher) preserves expert style.
    Correction loss adjusts weights away from collective failure patterns.
    """
    N = target.shape[-1]
    
    original_loss = -(
        torch.where(missing_mask, -torch.inf, pred).log_softmax(dim=-1).nan_to_num(0) 
        * torch.where(missing_mask, 0, target)
    ).sum(-1).sum(0)
    pred_logits = pred
    
    if failure_pattern is not None and correction_weight > 0:
        pred_weight = torch.where(missing_mask, -torch.inf, pred_logits).softmax(dim=-1)
        
        correction_loss = (pred_weight * failure_pattern).sum() * pred_weight.shape[-1]
        
        total_loss = kl_weight * original_loss + correction_weight * correction_loss
    else:
        total_loss = original_loss
    
    return total_loss


def train_with_retrospective_refinement(
    yearmons,
    backtest_start,
    backtest_end,
    date2ind,
    yearmon2indices,
    data,
    torch_data,
    torch_target_data,
    missing_mask,
    target_weight,
    model_cls,
    optimizer_cls,
    loss_fn,
    device: torch.device,
    window_size=0,
    num_repeat=100,
    num_epoch=100,
    lr=1e-2,
    stop_count=10,
    trading_cost=0.004,
    lagging=2,
    portfolio: str = 'maxsortino',
    enable_refinement: bool = True,
    consensus_threshold: float = 0.6,
    kl_weight: float = 0.7,
    correction_weight: float = 0.3,
    return_term_weight: float = 1.0,
    downside_term_weight: float = 0.0,
    downside_lookback_months: int = 12,
    downside_target: float = 0.0,
    utility_term_weight: float = 0.0,
    utility_floor: float = -0.95,
    utility_cap: float = 5.0,
    utility_score_clip: float = 3.0,
    utility_bad_month_weight: float = 2.0,
    utility_month_weight_cap: float = 3.0,
    cvar_term_weight: float = 0.0,
    cvar_alpha: float = 0.05,
    cvar_lookback_months: int = 1,
    cvar_score_clip: float = 3.0,
    cvar_bad_month_weight: float = 2.0,
    cvar_month_weight_cap: float = 3.0,
    first_round_weights: Optional[List[pd.DataFrame]] = None,
    first_round_logits: Optional[List[pd.DataFrame]] = None,
):
    """
    Two-stage retrospective refinement training.
    
    Stage 1: Normal training, collect predictions
    Diagnosis: Identify collective failure patterns (walk-forward, no leakage)
    Stage 2: Retrain with correction loss
    
    Args:
        enable_refinement: If False, just do normal training (for ablation)
        consensus_threshold: Fraction of seeds that must agree for collective failure
        kl_weight: Weight for original loss (style preservation)
        correction_weight: Weight for retrospective correction
        first_round_weights: Optional externally supplied stage-1 weight outputs
        first_round_logits: Optional externally supplied stage-1 logit outputs
    """
    if (first_round_weights is None) != (first_round_logits is None):
        raise ValueError('first_round_weights and first_round_logits must be provided together')

    base_input_size = torch_data.shape[1]
    num_item = len(data['log_tr'].columns)
    input_size = base_input_size
    num_features = base_input_size // num_item

    all_backtest_round1 = pd.DataFrame(index=data[backtest_start:backtest_end].index)

    if first_round_weights is not None:
        print("=" * 60)
        print("STAGE 1: Loading externally supplied first-round results")
        print("=" * 60)
        all_weights_round1 = first_round_weights
        all_logits_round1 = first_round_logits
        if len(all_weights_round1) != len(all_logits_round1):
            raise ValueError('external first-round weights/logits length mismatch')
        if len(all_weights_round1) == 0:
            raise ValueError('external first-round results are empty')
        num_repeat = len(all_weights_round1)
    else:
        print("=" * 60)
        print("STAGE 1: First-round training (baseline)")
        print("=" * 60)

        all_weights_round1 = []
        all_logits_round1 = []

        for rep in range(num_repeat):
            random.seed(rep)
            np.random.seed(rep)
            torch.manual_seed(rep)

            model = model_cls(input_size=input_size, num_output=num_item).to(device)
            optimizer = optimizer_cls(model.parameters(), lr=lr)

            backtest = pd.DataFrame(0.0, index=data[backtest_start:backtest_end].index, columns=['backtest'])
            weights = pd.DataFrame(0.0, index=yearmons, columns=data['log_tr'].columns).loc[backtest_start:backtest_end]
            logits = pd.DataFrame(0.0, index=yearmons, columns=data['log_tr'].columns).loc[backtest_start:backtest_end]

            prev_weight = None
            train_historical_missing_mask = missing_mask[yearmons[1]].clone()
            valid_historical_missing_mask = missing_mask[yearmons[2]].clone()
            test_historical_missing_mask = missing_mask[yearmons[3]].clone()

            for ind in (pbar := tqdm.tqdm(range(4, len(yearmons)), desc=f"Round1 Seed {rep}")):
                train_data = torch_data[[date2ind[x] for x in data.loc[:yearmons[ind-3]].index]]
                valid_data = torch_data[[date2ind[x] for x in data.loc[:yearmons[ind-2]].index]]
                test_data = torch_data[[date2ind[x] for x in data.loc[:yearmons[ind-1]].index]]

                if target_weight[yearmons[ind-2]].isnan().any() or target_weight[yearmons[ind-1]].isnan().any():
                    continue

                best_valid_loss = torch.inf
                count = 0
                for epoch in range(num_epoch):
                    model.train()

                    yearmon = yearmons[ind-2]
                    optimizer.zero_grad(set_to_none=True)
                    train_historical_missing_mask = train_historical_missing_mask & missing_mask[yearmon]
                    x = train_data[-window_size:].clone()
                    x[:, train_historical_missing_mask.repeat(num_features)] = 0
                    output = model(x, missing_mask=missing_mask[yearmon])
                    train_loss = loss_fn(output, target_weight[yearmon], missing_mask[yearmon])
                    train_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                    model.eval()
                    with torch.no_grad():
                        yearmon = yearmons[ind-1]
                        valid_historical_missing_mask = valid_historical_missing_mask & missing_mask[yearmon]
                        x = valid_data[-window_size:].clone()
                        x[:, valid_historical_missing_mask.repeat(num_features)] = 0
                        output = model(x, missing_mask=missing_mask[yearmon])
                        output = output[-1]
                        valid_loss = loss_fn(output, target_weight[yearmon], missing_mask[yearmon])

                        if best_valid_loss > valid_loss:
                            best_valid_loss = valid_loss
                            best_state_dict = model.state_dict()
                            count = 0
                        else:
                            count += 1

                        if count == stop_count:
                            break

                model.load_state_dict(best_state_dict)
                model.eval()

                with torch.no_grad():
                    yearmon = yearmons[ind]
                    test_historical_missing_mask = test_historical_missing_mask & missing_mask[yearmon]
                    x = test_data[-window_size:].clone()
                    x[:, test_historical_missing_mask.repeat(num_features)] = 0
                    output = model(x, missing_mask=missing_mask[yearmon])

                    predict_start, predict_end = yearmon2indices[yearmon]

                    if yearmon < backtest_start:
                        output = output[-1]
                        test_loss = loss_fn(output, target_weight[yearmon], missing_mask[yearmon])
                        pbar.set_description(f"Round1 Seed {rep} | Test loss: {test_loss:.6f}")
                    else:
                        output = output[-1]
                        pred_output = output
                        logits.loc[yearmon] = pred_output.cpu().numpy()
                        mask = missing_mask[yearmon]
                        cur_returns = torch.where(
                            mask.unsqueeze(0).expand(predict_end - predict_start + 1, -1),
                            torch.zeros_like(torch_target_data[predict_start + lagging:predict_end + 1 + lagging]),
                            torch_target_data[predict_start + lagging:predict_end + 1 + lagging]
                        ).exp()

                        weight = torch.where(missing_mask[yearmon], -torch.inf, pred_output).softmax(dim=-1)
                        weights.loc[yearmon] = weight.cpu().numpy()

                        tc = torch.as_tensor(trading_cost, dtype=weight.dtype, device=weight.device)
                        tradable_f = (~missing_mask[yearmon]).to(dtype=weight.dtype)
                        if prev_weight is None:
                            cost_vec = tradable_f * (weight * tc)
                        else:
                            cost_vec = tradable_f * ((weight - prev_weight).abs() * tc)
                        cur_returns[0] = cur_returns[0] - cost_vec

                        backtest_path = weight.reshape(1, -1).expand(predict_end - predict_start + 1, -1) * cur_returns
                        backtest_path = backtest_path.sum(-1).log()
                        backtest.loc[yearmon, 'backtest'] = backtest_path.cpu().numpy()

                        prev_weight = weight * cur_returns.prod(0)
                        prev_weight /= prev_weight.sum(-1)

                        test_return = backtest.loc[:yearmon].apply(np.exp).prod().iloc[0]
                        test_sharpe = (backtest.loc[:yearmon].mean() * np.sqrt(252) / backtest.loc[:yearmon].std()).iloc[0]
                        pbar.set_description(f"Round1 Seed {rep} | Ret: {test_return:.3f}, Sharpe: {test_sharpe:.3f}")

            backtest = backtest.shift(lagging).fillna(0)
            all_backtest_round1 = pd.concat((all_backtest_round1, backtest), axis=1)
            all_weights_round1.append(weights)
            all_logits_round1.append(logits)

    if not enable_refinement:
        print("\nRefinement disabled. Returning Round 1 results.")
        return all_backtest_round1, all_weights_round1, all_logits_round1
    
    print("\n" + "=" * 60)
    print("DIAGNOSIS: Analyzing collective failure patterns")
    print("=" * 60)
    
    failure_patterns = _diagnose_collective_failure_patterns(
        first_round_weights=all_weights_round1,
        first_round_logits=all_logits_round1,
        torch_target_data=torch_target_data,
        yearmon2indices=yearmon2indices,
        missing_mask=missing_mask,
        yearmons=yearmons,
        backtest_start=backtest_start,
        lagging=lagging,
        trading_cost=trading_cost,
        device=device,
        target_weight=target_weight,
        portfolio=portfolio,
        consensus_threshold=consensus_threshold,
        return_term_weight=return_term_weight,
        downside_term_weight=downside_term_weight,
        downside_lookback_months=downside_lookback_months,
        downside_target=downside_target,
        utility_term_weight=utility_term_weight,
        utility_floor=utility_floor,
        utility_cap=utility_cap,
        utility_score_clip=utility_score_clip,
        utility_bad_month_weight=utility_bad_month_weight,
        utility_month_weight_cap=utility_month_weight_cap,
        cvar_term_weight=cvar_term_weight,
        cvar_alpha=cvar_alpha,
        cvar_lookback_months=cvar_lookback_months,
        cvar_score_clip=cvar_score_clip,
        cvar_bad_month_weight=cvar_bad_month_weight,
        cvar_month_weight_cap=cvar_month_weight_cap,
    )
    
    print(f"Diagnosed {len(failure_patterns)} months with collective patterns.")
    
    print("\n" + "=" * 60)
    print("STAGE 2: Retrospective refinement training")
    print("=" * 60)
    
    all_backtest_round2 = pd.DataFrame(index=data[backtest_start:backtest_end].index)
    all_weights_round2 = []
    all_logits_round2 = []

    for rep in range(num_repeat):
        random.seed(rep)
        np.random.seed(rep)
        torch.manual_seed(rep)

        model = model_cls(input_size=input_size, num_output=num_item).to(device)
        optimizer = optimizer_cls(model.parameters(), lr=lr)

        backtest = pd.DataFrame(0.0, index=data[backtest_start:backtest_end].index, columns=['backtest'])
        weights = pd.DataFrame(0.0, index=yearmons, columns=data['log_tr'].columns).loc[backtest_start:backtest_end]
        logits = pd.DataFrame(0.0, index=yearmons, columns=data['log_tr'].columns).loc[backtest_start:backtest_end]

        prev_weight = None
        train_historical_missing_mask = missing_mask[yearmons[1]].clone()
        valid_historical_missing_mask = missing_mask[yearmons[2]].clone()
        test_historical_missing_mask = missing_mask[yearmons[3]].clone()

        for ind in (pbar := tqdm.tqdm(range(4, len(yearmons)), desc=f"Round2 Seed {rep}")):
            train_data = torch_data[[date2ind[x] for x in data.loc[:yearmons[ind-3]].index]]
            valid_data = torch_data[[date2ind[x] for x in data.loc[:yearmons[ind-2]].index]]
            test_data = torch_data[[date2ind[x] for x in data.loc[:yearmons[ind-1]].index]]

            if target_weight[yearmons[ind-2]].isnan().any() or target_weight[yearmons[ind-1]].isnan().any():
                continue

            best_valid_loss = torch.inf
            count = 0
            for epoch in range(num_epoch):
                model.train()

                yearmon = yearmons[ind-2]
                optimizer.zero_grad(set_to_none=True)
                train_historical_missing_mask = train_historical_missing_mask & missing_mask[yearmon]
                x = train_data[-window_size:].clone()
                x[:, train_historical_missing_mask.repeat(num_features)] = 0
                output = model(x, missing_mask=missing_mask[yearmon])
                
                current_pattern = _get_causal_retrospective_pattern(
                    failure_patterns=failure_patterns,
                    current_yearmon=yearmon,
                    current_missing_mask=missing_mask[yearmon],
                )
                
                train_loss = _retrospective_correction_loss(
                    pred=output,
                    target=target_weight[yearmon],
                    missing_mask=missing_mask[yearmon],
                    failure_pattern=current_pattern,
                    kl_weight=kl_weight,
                    correction_weight=correction_weight,
                )
                train_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                model.eval()
                with torch.no_grad():
                    yearmon = yearmons[ind-1]
                    valid_historical_missing_mask = valid_historical_missing_mask & missing_mask[yearmon]
                    x = valid_data[-window_size:].clone()
                    x[:, valid_historical_missing_mask.repeat(num_features)] = 0
                    output = model(x, missing_mask=missing_mask[yearmon])
                    output = output[-1]
                    
                    current_pattern_val = _get_causal_retrospective_pattern(
                        failure_patterns=failure_patterns,
                        current_yearmon=yearmon,
                        current_missing_mask=missing_mask[yearmon],
                    )
                    
                    valid_loss = _retrospective_correction_loss(
                        pred=output,
                        target=target_weight[yearmon],
                        missing_mask=missing_mask[yearmon],
                        failure_pattern=current_pattern_val,
                        kl_weight=kl_weight,
                        correction_weight=correction_weight,
                    )

                    if best_valid_loss > valid_loss:
                        best_valid_loss = valid_loss
                        best_state_dict = model.state_dict()
                        count = 0
                    else:
                        count += 1

                    if count == stop_count:
                        break

            model.load_state_dict(best_state_dict)
            model.eval()

            with torch.no_grad():
                yearmon = yearmons[ind]
                test_historical_missing_mask = test_historical_missing_mask & missing_mask[yearmon]
                x = test_data[-window_size:].clone()
                x[:, test_historical_missing_mask.repeat(num_features)] = 0
                output = model(x, missing_mask=missing_mask[yearmon])

                predict_start, predict_end = yearmon2indices[yearmon]

                if yearmon < backtest_start:
                    output = output[-1]
                    current_pattern_test = _get_causal_retrospective_pattern(
                        failure_patterns=failure_patterns,
                        current_yearmon=yearmon,
                        current_missing_mask=missing_mask[yearmon],
                    )
                    test_loss = _retrospective_correction_loss(
                        pred=output,
                        target=target_weight[yearmon],
                        missing_mask=missing_mask[yearmon],
                        failure_pattern=current_pattern_test,
                        kl_weight=kl_weight,
                        correction_weight=correction_weight,
                    )
                    pbar.set_description(f"Round2 Seed {rep} | Test loss: {test_loss:.6f}")
                else:
                    output = output[-1]
                    pred_output = output
                    logits.loc[yearmon] = pred_output.cpu().numpy()
                    mask = missing_mask[yearmon]
                    cur_returns = torch.where(
                        mask.unsqueeze(0).expand(predict_end - predict_start + 1, -1),
                        torch.zeros_like(torch_target_data[predict_start + lagging:predict_end + 1 + lagging]),
                        torch_target_data[predict_start + lagging:predict_end + 1 + lagging]
                    ).exp()

                    weight = torch.where(missing_mask[yearmon], -torch.inf, pred_output).softmax(dim=-1)
                    weights.loc[yearmon] = weight.cpu().numpy()

                    tc = torch.as_tensor(trading_cost, dtype=weight.dtype, device=weight.device)
                    tradable_f = (~missing_mask[yearmon]).to(dtype=weight.dtype)
                    if prev_weight is None:
                        cost_vec = tradable_f * (weight * tc)
                    else:
                        cost_vec = tradable_f * ((weight - prev_weight).abs() * tc)
                    cur_returns[0] = cur_returns[0] - cost_vec

                    backtest_path = weight.reshape(1, -1).expand(predict_end - predict_start + 1, -1) * cur_returns
                    backtest_path = backtest_path.sum(-1).log()
                    backtest.loc[yearmon, 'backtest'] = backtest_path.cpu().numpy()

                    prev_weight = weight * cur_returns.prod(0)
                    prev_weight /= prev_weight.sum(-1)

                    test_return = backtest.loc[:yearmon].apply(np.exp).prod().iloc[0]
                    test_sharpe = (backtest.loc[:yearmon].mean() * np.sqrt(252) / backtest.loc[:yearmon].std()).iloc[0]
                    pbar.set_description(f"Round2 Seed {rep} | Ret: {test_return:.3f}, Sharpe: {test_sharpe:.3f}")

        backtest = backtest.shift(lagging).fillna(0)
        all_backtest_round2 = pd.concat((all_backtest_round2, backtest), axis=1)
        all_weights_round2.append(weights)
        all_logits_round2.append(logits)

    print("\n" + "=" * 60)
    print("Two-stage retrospective refinement completed!")
    print("=" * 60)
    
    return all_backtest_round2, all_weights_round2, all_logits_round2

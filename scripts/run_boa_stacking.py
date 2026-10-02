import math

import numpy as np


def stable_softmax(logits):
    logits = np.asarray(logits, dtype=float)
    logits = np.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
    logits = logits - np.max(logits)
    raw = np.exp(logits)
    total = float(raw.sum())
    if total <= 1e-12 or not np.isfinite(total):
        return np.full(len(raw), 1.0 / len(raw), dtype=float)
    return raw / total


def cap_simplex(weights, cap):
    weights = np.asarray(weights, dtype=float)
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    weights[weights < 0.0] = 0.0
    if weights.sum() <= 1e-12:
        weights = np.full(len(weights), 1.0 / len(weights), dtype=float)
    else:
        weights = weights / weights.sum()
    if cap <= 0.0 or cap >= 1.0:
        return weights
    if cap * len(weights) < 1.0 - 1e-12:
        return weights
    out = weights.copy()
    for _ in range(100):
        over = out > cap
        if not over.any():
            break
        excess = float((out[over] - cap).sum())
        out[over] = cap
        under = ~over
        if not under.any() or out[under].sum() <= 1e-12:
            break
        out[under] += excess * out[under] / out[under].sum()
    return out / max(out.sum(), 1e-12)


def boa_prob(prior, regret, variation, eta, second_order_scale, max_weight):
    prior = np.asarray(prior, dtype=float)
    prior = prior / max(float(prior.sum()), 1e-12)
    logits = np.log(prior + 1e-12) + float(eta) * regret - float(second_order_scale) * (float(eta) ** 2) * variation
    return cap_simplex(stable_softmax(logits), max_weight)


def log_loss_from_return(ret, downside_penalty=0.0):
    ret = float(ret)
    loss = -math.log(max(1.0 + ret, 1e-12))
    if downside_penalty and downside_penalty > 0.0:
        loss += float(downside_penalty) * max(-ret, 0.0) ** 2
    return loss


def auto_eta(n_options, horizon):
    return math.sqrt(2.0 * math.log(max(n_options, 2)) / max(float(horizon), 1.0))

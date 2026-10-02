import argparse
import os
import pickle
import sys
import pathlib
import numpy as np
import pandas as pd
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import preprocess_backtest_data, get_yearmons


def safe_norm(w: np.ndarray, mask_valid: np.ndarray) -> np.ndarray:
    w = np.where(mask_valid, w, 0.0)
    s = w.sum()
    if s > 1e-12:
        return w / s
    cnt = mask_valid.sum()
    if cnt > 0:
        w = np.where(mask_valid, 1.0 / cnt, 0.0)
        return w
    return w


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = (p + eps) / max(p.sum(), eps)
    q = (q + eps) / max(q.sum(), eps)
    return float(np.sum(p * (np.log(p) - np.log(q))))


def topk_overlap(p: np.ndarray, q: np.ndarray, k: int) -> float:
    k = int(min(k, p.size, q.size))
    if k <= 0:
        return float('nan')
    idx_p = np.argsort(-p)[:k]
    idx_q = np.argsort(-q)[:k]
    inter = len(set(idx_p.tolist()).intersection(set(idx_q.tolist())))
    return inter / float(k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--universe', type=str, required=True)
    ap.add_argument('--weights_pkl', type=str, required=True)
    ap.add_argument('--teacher_pkl', type=str, required=True)
    ap.add_argument('--start', type=str, default='2010-01')
    ap.add_argument('--end', type=str, default='2025-05')
    ap.add_argument('--lag', type=int, default=2)
    ap.add_argument('--cost', type=float, default=0.004)
    ap.add_argument('--out_prefix', type=str, required=True)
    ap.add_argument('--ensemble_mode', type=str, default='returns_mean', choices=['returns_mean','weights_mean'])
    args = ap.parse_args()

    device = torch.device('cpu')

    yearmons, data, missing_mask = preprocess_backtest_data(os.path.join('data', f'{args.universe}.csv'), device)
    tickers = list(data['log_tr'].columns)
    date2ind = {x: ind for ind, x in enumerate(data.index)}
    ym2idx = {ym: (date2ind[data.loc[ym].index[0]], date2ind[data.loc[ym].index[-1]]) for ym in get_yearmons(data)}

    months = [m for m in get_yearmons(data) if args.start <= m <= args.end]

    with open(args.weights_pkl, 'rb') as f:
        all_weights = pickle.load(f)

    W = {}
    for m in months:
        stack = []
        for df in all_weights:
            if m in df.index:
                vals = df.loc[m].reindex(tickers).astype(float).values
                stack.append(vals)
        if len(stack) == 0:
            continue
        w_avg = np.nanmean(np.stack(stack, axis=0), axis=0)
        w_avg = np.nan_to_num(w_avg, nan=0.0, posinf=0.0, neginf=0.0)
        w_avg[w_avg < 0] = 0.0
        W[m] = w_avg

    def compute_month_log(seg: np.ndarray, mm: np.ndarray, w: np.ndarray, prev_w: np.ndarray) -> tuple:
        seg = np.where(mm.reshape(1, -1), 0.0, seg)
        seg = np.nan_to_num(seg, nan=0.0, posinf=0.0, neginf=0.0)
        cur_ret = np.exp(seg)
        valid = (~mm)
        w = np.nan_to_num(w.copy(), nan=0.0, posinf=0.0, neginf=0.0)
        w[w < 0] = 0.0
        if float(np.sum(np.where(valid, w, 0.0))) <= 1e-12:
            return 0.0, prev_w
        w = safe_norm(w, valid)
        if cur_ret.shape[0] > 0:
            if prev_w is None:
                cur_ret[0, :] = cur_ret[0, :] - (w * float(args.cost))
            else:
                cur_ret[0, :] = cur_ret[0, :] - (np.abs(w - prev_w) * float(args.cost))
        mix = (cur_ret * w.reshape(1, -1)).sum(axis=1)
        mix = np.nan_to_num(mix, nan=1.0, posinf=1e12, neginf=1e-12)
        mix = np.clip(mix, 1e-12, None)
        month_log = float(np.log(mix).sum())
        prod_vec = cur_ret.prod(axis=0)
        next_prev_w = safe_norm(w * prod_vec, valid)
        return month_log, next_prev_w

    out_months = []
    logs = []
    equity_vals = []
    eq = 1.0
    if args.ensemble_mode == 'weights_mean':
        prev_w = None
        for m in months:
            if m not in W:
                continue
            start_idx, end_idx = ym2idx[m]
            t0 = start_idx + int(args.lag)
            t1 = min(end_idx + int(args.lag), len(data.index) - 1)
            if t0 > t1:
                continue
            seg = data['log_tr'].iloc[t0:t1+1].values.astype(float)
            mm = missing_mask[m].cpu().numpy().astype(bool) if m in missing_mask else np.zeros((len(tickers),), dtype=bool)
            month_log, prev_w = compute_month_log(seg, mm, W[m], prev_w)
            if not np.isfinite(month_log):
                continue
            logs.append(month_log)
            eq *= float(np.exp(month_log))
            equity_vals.append(eq)
            out_months.append(m)
    else:
        prev_ws = [None for _ in all_weights]
        for m in months:
            start_idx, end_idx = ym2idx[m]
            t0 = start_idx + int(args.lag)
            t1 = min(end_idx + int(args.lag), len(data.index) - 1)
            if t0 > t1:
                continue
            seg = data['log_tr'].iloc[t0:t1+1].values.astype(float)
            mm = missing_mask[m].cpu().numpy().astype(bool) if m in missing_mask else np.zeros((len(tickers),), dtype=bool)
            month_logs = []
            for i, df in enumerate(all_weights):
                if m not in df.index:
                    continue
                w_i = df.loc[m].reindex(tickers).astype(float).values
                ml, prev_ws[i] = compute_month_log(seg, mm, w_i, prev_ws[i])
                if np.isfinite(ml):
                    month_logs.append(ml)
            if len(month_logs) == 0:
                continue
            month_log = float(np.mean(month_logs))
            logs.append(month_log)
            eq *= float(np.exp(month_log))
            equity_vals.append(eq)
            out_months.append(m)

    os.makedirs(os.path.dirname(args.out_prefix), exist_ok=True)
    df_eq = pd.DataFrame({'month': out_months, 'equity': equity_vals})
    df_eq.to_csv(args.out_prefix + '_equity.csv', index=False)

    logs_arr = np.array(logs, dtype=float)
    eq_arr = np.array(equity_vals, dtype=float)
    if logs_arr.size == 0 and eq_arr.size > 1:
        logs_arr = np.log(eq_arr[1:] / np.clip(eq_arr[:-1], 1e-12, None))
    if logs_arr.size > 0:
        ann_ret = float(np.exp(logs_arr.mean() * 12))
        ann_vol = float(logs_arr.std() * np.sqrt(12))
        sharpe = float((logs_arr.mean() * np.sqrt(12)) / (logs_arr.std() + 1e-12))
        downside = np.minimum(logs_arr, 0.0)
        downside_std = float(np.sqrt(np.mean(downside * downside)))
        sortino = float((logs_arr.mean() * np.sqrt(12)) / (downside_std + 1e-12))
    else:
        ann_ret = ann_vol = sharpe = sortino = float('nan')
    if eq_arr.size > 0:
        max_dd = float((eq_arr / np.maximum.accumulate(eq_arr) - 1.0).min())
    else:
        max_dd = float('nan')
    pd.DataFrame([{
        'total_return': float(equity_vals[-1]) if len(equity_vals) else 1.0,
        'annualized_return': ann_ret,
        'annualized_vol': ann_vol,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_drawdown': max_dd,
    }]).to_csv(args.out_prefix + '_summary.csv', index=False)

    with open(args.teacher_pkl, 'rb') as f:
        teacher_obj = pickle.load(f)
    rows = []
    ks = [10, 20, 50]
    for m in out_months:
        if m not in W:
            continue
        if m not in teacher_obj:
            continue
        mm = missing_mask[m].cpu().numpy().astype(bool) if m in missing_mask else np.zeros((len(tickers),), dtype=bool)
        p = teacher_obj[m].detach().cpu().numpy().astype(float)
        q = W[m].astype(float)
        if float(np.sum(q)) <= 1e-12:
            continue
        valid = ~mm
        if valid.sum() <= 0:
            continue
        p = safe_norm(p, valid)
        q = safe_norm(q, valid)
        rec = {
            'month': m,
            'kl_teacher_to_student': kl_divergence(p, q),
            'kl_student_to_teacher': kl_divergence(q, p),
        }
        for k in ks:
            rec[f'top{k}_overlap'] = topk_overlap(p, q, k)
        rows.append(rec)
    df_ag = pd.DataFrame(rows).sort_values('month')
    df_ag.to_csv(args.out_prefix + '_agreement.csv', index=False)

    if len(df_ag) > 0:
        avg = df_ag.drop(columns=['month']).mean(numeric_only=True)
        print('Averages:')
        for k, v in avg.items():
            print(f'{k}: {v:.6f}')
        print('Saved to', args.out_prefix + '_equity.csv', 'and', args.out_prefix + '_summary.csv', 'and', args.out_prefix + '_agreement.csv')
    else:
        print('No agreement rows computed. Check months alignment and inputs.')


if __name__ == '__main__':
    main()


import argparse
import hashlib
import pickle
import os

import torch

from src.utils import preprocess_data
from src.model import lstm, trf, mamba
from src.loss import loss_fn
from src.retrospective_regularizers import (
    get_available_regularizer_presets,
    resolve_portfolio_regularizer_kwargs,
)
from src.train_retrospective import train_with_retrospective_refinement
from sophia import SophiaG

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def _compact_output_tag(tag: str, max_len: int = 48) -> str:
    value = str(tag).strip()
    if not value:
        return ''
    safe = ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in value).strip('_')
    if not safe:
        return ''
    if len(safe) <= max_len:
        return safe
    digest = hashlib.sha1(safe.encode('utf-8')).hexdigest()[:10]
    head_len = max(max_len - len(digest) - 1, 8)
    return f'{safe[:head_len]}_{digest}'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--universe', default='universe1', type=str)
    parser.add_argument('-p', '--portfolio', default='maxsortino', choices=['maxsharpe', 'maxsortino', 'maxlogutility', 'mincvar'])
    parser.add_argument('-o', '--model', default='mamba', choices=['lstm', 'trf', 'mamba'])
    parser.add_argument('-s', '--train_start', default='2010-01')
    parser.add_argument('-b', '--backtest_start', default='2019-11')
    parser.add_argument('-l', '--lr', default=1e-4, type=float)
    parser.add_argument('-w', '--window_size', default=252, type=int)
    parser.add_argument('-r', '--num_repeat', default=100, type=int)
    parser.add_argument('-e', '--num_epoch', default=50, type=int)
    parser.add_argument('-t', '--trading_cost', default=0.004, type=float)
    parser.add_argument('-g', '--lagging', default=2, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--end_date', default='', type=str)
    parser.add_argument('--out_tag', default='', type=str)
    parser.add_argument('--save_dir', default='result', type=str)
    parser.add_argument('--regularizer_preset', default=None, type=str,
                        help='Override the default single preset for this objective; leave unset for the main preset')

    parser.add_argument('--enable_refinement', action='store_true',
                        help='Enable second-stage retrospective refinement (default: False for ablation)')
    parser.add_argument('--first_round_weights_path', default='', type=str,
                        help='Optional path to external stage-1 all_weights.pkl used to extract failure experience')
    parser.add_argument('--first_round_logits_path', default='', type=str,
                        help='Optional path to external stage-1 all_logits.pkl used to extract failure experience')

    args = parser.parse_args()

    available_presets = get_available_regularizer_presets(args.portfolio)
    regularizer_preset = args.regularizer_preset if args.regularizer_preset is not None else available_presets[0]
    if regularizer_preset not in available_presets:
        raise ValueError(
            f'Unknown retrospective regularizer preset {regularizer_preset} for {args.portfolio}. '
            f'Available presets: {", ".join(available_presets)}'
        )

    resolved_regularizer_kwargs = resolve_portfolio_regularizer_kwargs(
        portfolio=args.portfolio,
        preset=regularizer_preset,
    )

    device_str = str(args.device).lower().strip()
    if device_str == 'cuda' and not torch.cuda.is_available():
        device_str = 'cpu'
    device = torch.device(device_str)

    if bool(args.first_round_weights_path) != bool(args.first_round_logits_path):
        raise ValueError('first_round_weights_path and first_round_logits_path must be provided together')

    with open(f'result/{args.universe}_weights_{args.portfolio}.pkl', 'rb') as f:
        target_weight = pickle.load(f)

    for key in target_weight.keys():
        target_weight[key] = target_weight[key].to(device)

    yearmons, data, torch_data, torch_target_data, missing_mask = preprocess_data(
        f'data/{args.universe}.csv',
        args.train_start,
        device,
        end_date=args.end_date,
    )

    backtest_start = args.backtest_start
    backtest_end = yearmons[-1]
    date2ind = {x: ind for ind, x in enumerate(data.index)}
    yearmon2indices = {x: (date2ind[data.loc[x].index[0]], date2ind[data.loc[x].index[-1]]) for ind, x in enumerate(yearmons)}

    model_cls = {'lstm': lstm, 'trf': trf, 'mamba': mamba}[args.model]

    first_round_weights = None
    first_round_logits = None
    first_round_source_tag = ''
    if args.first_round_weights_path:
        with open(args.first_round_weights_path, 'rb') as f:
            first_round_weights = pickle.load(f)
        with open(args.first_round_logits_path, 'rb') as f:
            first_round_logits = pickle.load(f)
        first_round_source_tag = os.path.splitext(os.path.basename(args.first_round_logits_path))[0]

    print("\n" + "=" * 80)
    print("RETROSPECTIVE REFINEMENT TRAINING")
    print("=" * 80)
    print(f"Universe: {args.universe}")
    print(f"Model: {args.model}")
    print(f"Portfolio: {args.portfolio}")
    print(f"Backtest: {backtest_start} -> {backtest_end}")
    print(f"Seeds: {args.num_repeat}")
    print(f"Refinement enabled: {args.enable_refinement}")
    print(f"Regularizer preset: {regularizer_preset}")
    if first_round_source_tag:
        print(f"External first-round source: {first_round_source_tag}")
    if args.enable_refinement:
        for field_name, field_value in resolved_regularizer_kwargs.items():
            print(f"  - {field_name}: {field_value}")
    print("=" * 80 + "\n")

    all_backtest, all_weights, all_logits = train_with_retrospective_refinement(
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
        SophiaG,
        loss_fn,
        device,
        window_size=args.window_size,
        num_repeat=args.num_repeat,
        num_epoch=args.num_epoch,
        lr=args.lr,
        trading_cost=args.trading_cost,
        lagging=args.lagging,
        portfolio=args.portfolio,
        enable_refinement=args.enable_refinement,
        **resolved_regularizer_kwargs,
        first_round_weights=first_round_weights,
        first_round_logits=first_round_logits,
    )

    refinement_suffix = '_retro' if args.enable_refinement else '_baseline'
    source_tag_compact = _compact_output_tag(first_round_source_tag, max_len=40)
    source_suffix = f'_from_{source_tag_compact}' if source_tag_compact else ''
    preset_suffix = f'_{_compact_output_tag(regularizer_preset, max_len=32)}'
    out_tag_compact = _compact_output_tag(args.out_tag, max_len=48)
    out_tag_suffix = f'_{out_tag_compact}' if out_tag_compact else preset_suffix
    os.makedirs(args.save_dir, exist_ok=True)
    artifact_stem = f'{args.universe}_{args.portfolio}_{args.model}_sophia_{args.window_size}{refinement_suffix}{source_suffix}{out_tag_suffix}'
    projected_weights_path = os.path.abspath(os.path.join(args.save_dir, artifact_stem + '_all_weights.pkl'))
    if len(projected_weights_path) >= 240:
        stem_digest = hashlib.sha1(artifact_stem.encode('utf-8')).hexdigest()[:10]
        compact_parts = [
            args.universe,
            args.portfolio,
            args.model,
            'sophia',
            str(args.window_size),
            'retro' if args.enable_refinement else 'baseline',
        ]
        if source_tag_compact:
            compact_parts.append('src')
            compact_parts.append(hashlib.sha1(first_round_source_tag.encode('utf-8')).hexdigest()[:8])
        if out_tag_compact:
            compact_parts.append(out_tag_compact[:24])
        else:
            compact_parts.append(_compact_output_tag(regularizer_preset, max_len=24))
        compact_parts.append(stem_digest)
        artifact_stem = '_'.join(compact_parts)
    common_path = os.path.join(args.save_dir, artifact_stem)

    print(f"\nSaving results to: {common_path}_*")

    with open(common_path + '_all_weights.pkl', 'wb') as f:
        pickle.dump(all_weights, f)
    with open(common_path + '_all_logits.pkl', 'wb') as f:
        pickle.dump(all_logits, f)
    with open(common_path + '_all_backtest.pkl', 'wb') as f:
        pickle.dump(all_backtest, f)
    
    print("\nTraining completed successfully!")
    print(f"Output files:")
    print(f"  - {common_path}_all_weights.pkl")
    print(f"  - {common_path}_all_logits.pkl")
    print(f"  - {common_path}_all_backtest.pkl")

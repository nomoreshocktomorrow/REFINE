from dataclasses import asdict, dataclass, fields, replace
from typing import Dict, Mapping, Optional, Tuple


@dataclass(frozen=True)
class RetrospectiveRegularizerSpec:
    consensus_threshold: float = 0.6
    kl_weight: float = 0.7
    correction_weight: float = 0.3
    return_term_weight: float = 1.0
    downside_term_weight: float = 0.0
    downside_lookback_months: int = 12
    downside_target: float = 0.0
    utility_term_weight: float = 0.0
    utility_floor: float = -0.95
    utility_cap: float = 5.0
    utility_score_clip: float = 3.0
    utility_bad_month_weight: float = 2.0
    utility_month_weight_cap: float = 3.0
    cvar_term_weight: float = 0.0
    cvar_alpha: float = 0.05
    cvar_lookback_months: int = 1
    cvar_score_clip: float = 3.0
    cvar_bad_month_weight: float = 2.0
    cvar_month_weight_cap: float = 3.0


RETROSPECTIVE_REGULARIZER_FIELDS: Tuple[str, ...] = tuple(field.name for field in fields(RetrospectiveRegularizerSpec))


_PORTFOLIO_PRESET_SPECS: Dict[str, Dict[str, RetrospectiveRegularizerSpec]] = {
    'maxsharpe': {
        'best': RetrospectiveRegularizerSpec(
            consensus_threshold=0.65,
            kl_weight=0.9,
            correction_weight=0.015,
        ),
    },
    'maxsortino': {
        'best': RetrospectiveRegularizerSpec(
            consensus_threshold=0.70,
            kl_weight=1.0,
            correction_weight=0.002,
            downside_term_weight=0.5,
            downside_lookback_months=12,
            downside_target=0.0,
        ),
    },
    'maxlogutility': {
        'best': RetrospectiveRegularizerSpec(
            consensus_threshold=0.75,
            kl_weight=1.0,
            correction_weight=0.005,
            return_term_weight=1.0,
            utility_term_weight=1.0,
            utility_floor=-0.95,
            utility_cap=5.0,
            utility_score_clip=3.0,
            utility_bad_month_weight=2.0,
            utility_month_weight_cap=3.0,
        ),
    },
    'mincvar': {
        'cvar_tail_focus': RetrospectiveRegularizerSpec(
            consensus_threshold=0.70,
            kl_weight=1.0,
            correction_weight=0.005,
            return_term_weight=0.25,
            cvar_term_weight=1.0,
            cvar_alpha=0.05,
            cvar_lookback_months=6,
            cvar_score_clip=3.0,
            cvar_bad_month_weight=2.0,
            cvar_month_weight_cap=3.0,
        ),
    },
}


def _default_preset_name(preset_specs: Dict[str, RetrospectiveRegularizerSpec]) -> str:
    return next(iter(preset_specs))


def get_available_regularizer_presets(portfolio: str) -> Tuple[str, ...]:
    preset_specs = _PORTFOLIO_PRESET_SPECS.get(portfolio)
    if preset_specs is None:
        raise ValueError(f'Unknown portfolio for retrospective regularizer preset resolution: {portfolio}')
    return tuple(preset_specs.keys())


def get_portfolio_regularizer_spec(portfolio: str, preset: Optional[str] = None) -> RetrospectiveRegularizerSpec:
    preset_specs = _PORTFOLIO_PRESET_SPECS.get(portfolio)
    if preset_specs is None:
        raise ValueError(f'Unknown portfolio for retrospective regularizer preset resolution: {portfolio}')
    if preset is None:
        preset = _default_preset_name(preset_specs)
    if preset not in preset_specs:
        available = ', '.join(preset_specs.keys())
        raise ValueError(f'Unknown retrospective regularizer preset {preset} for {portfolio}. Available presets: {available}')
    return preset_specs[preset]


def resolve_portfolio_regularizer_spec(
    portfolio: str,
    preset: Optional[str] = None,
    overrides: Optional[Mapping[str, object]] = None,
) -> RetrospectiveRegularizerSpec:
    spec = get_portfolio_regularizer_spec(portfolio=portfolio, preset=preset)
    if overrides:
        valid_overrides = {k: v for k, v in overrides.items() if k in RETROSPECTIVE_REGULARIZER_FIELDS}
        if valid_overrides:
            spec = replace(spec, **valid_overrides)
    return spec


def resolve_portfolio_regularizer_kwargs(
    portfolio: str,
    preset: Optional[str] = None,
    overrides: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    return asdict(resolve_portfolio_regularizer_spec(portfolio=portfolio, preset=preset, overrides=overrides))

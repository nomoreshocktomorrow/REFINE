# REFINE: Reflection-Enhanced Financial portfolio learning via Incremental Network Ensemble

This repository contains the official implementation of the paper "REFINE: Reflection-Enhanced Financial portfolio learning via Incremental Network Ensemble".

## Overview

REFINE is a decision-focused supervised learning and ensemble framework for portfolio optimization under non-stationarity. It trains a heterogeneous pool of objective-specific deep experts from finance-motivated target allocations, regularizes them with distilled market experience, and combines their decisions through feature-guided online fusion. The framework is validated via rolling out-of-sample backtests on five fixed U.S. equity universes.

## Key Innovations

REFINE addresses portfolio construction under non-stationarity through three key mechanisms:

1. **Multi-objective expert pool**: Four classical criteria - maximum compounded growth, maximum Sharpe, maximum Sortino, and minimum CVaR - supervise diverse deep experts, avoiding objective drift as market regimes rotate.
2. **Market-experience regularization**: A second-stage retrospective correction distills realized misallocation patterns from independently seeded replicas and injects them as an objective-aligned regularizer.
3. **Feature-guided online fusion**: A multiplicative-weights aggregation scheme combines expert groups, initialized with a mean-reverting prior over recent compounded performance and updated using exponentially discounted first- and second-order relative advantage.

## Architecture

The REFINE framework consists of three main stages:

1. **Target Allocation Construction**: For each rolling cycle, objective-implied target allocations are solved under the four criteria.
2. **Retrospective Expert Refinement**: LSTM, Transformer, and Mamba backbones are trained on the targets and refined with realized misallocation signals.
3. **Online Fusion and Backtest**: Expert weights are aggregated online and evaluated with monthly rebalancing, 0.4% proportional transaction costs, and a two-day execution lag.

## Datasets

The model is evaluated on five fixed U.S. equity universes:

- **Large-cap Stock**: 23 names, training from Jan 2010, out-of-sample from Nov 2019 to May 2025.
- **Range-Bounded Stock**: 84 names, same period.
- **High-Yield Stock**: 20 names, same period.
- **Defensive Sector**: 29 names, same period.
- **NASDAQ Top100**: 100 largest NASDAQ constituents, same period.

Daily VOHLC (Volume, Open, High, Low, Close) data for the four hand-picked universes are sourced from the iFinD financial data terminal.

## Results

REFINE outperforms classical allocation rules, prediction-focused learning methods, and prior decision-focused learning baselines across the five universes. It achieves the highest cumulative return in three universes and the best risk-adjusted performance in four universes, while the released fusion coefficients improve ex post interpretability of allocations.

The curated package provides the main result artifacts:

- `results/Final/`: fused final portfolio weights named `Dataset__final.pkl` / `.csv`.
- `results/Phase I/`: first-round expert weights named `Dataset_Objective_Model_PhaseI.pkl`.
- `results/Phase II/`: retrospective (second-round) expert weights named `Dataset_Objective_Model_PhaseII.pkl`.



# Implementation Plan: Institutional Transparency & Universe Accounting for LightGBM Alpha158 Training Pipeline

## Context & Objectives
In compliance with .team-code/requirements.md and the governance mandates of **The Profitable Stock Trader** and **The Institutional Hedge Fund Manager**, this plan refactors scripts/train_alpha158_lightgbm.py to eliminate black-box training and silent data dropping in Microsoft Qlib.

## Requirements
1. **Universe & Physical Feature Storage Audit**:
   - Pre-flight discovery cross-checking ussell1000.txt against qlib_dir / "features" / <symbol.lower()>.
   - Identification of targeted, valid (with binary files > 4 bytes), missing, corrupted, and delisted tickers.
2. **Dataset & Training Dimensions Accounting**:
   - Extraction of segment dimensions (	rain, alid, 	est) without memory bloat using label projections (DK_L / DK_I).
   - Reporting observation rows, date spans, active ticker counts, and daily cross-sectional breadth statistics (min, mean, median, max).
3. **Institutional Model Performance & Factor Attribution**:
   - Calculation of Pearson IC, Spearman Rank IC, daily ICIR, and **Annualized ICIR** ($\times \sqrt{252}$).
   - Runtime extraction of mathematical factor formulas via Alpha158DL.get_feature_config().
   - Semantic factor mapping explaining the behavioral / economic rationale of top alpha drivers.
4. **Institutional Summary Banner**:
   - Comprehensive multi-section terminal output with Windows UTF-8 stream protection.

## Functions to Implement
- udit_universe_and_features
- udit_dataset_segments
- esolve_factor_attribution
- print_institutional_summary_banner

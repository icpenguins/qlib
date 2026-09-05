# print_institutional_summary_banner Specification

## Purpose
Outputs an institutional-grade, multi-section terminal summary banner designed for quantitative asset managers, CIOs, and proprietary traders.

## Sections Rendered
1. **Universe Discovery & Ticker Accounting**: Targeted instruments, valid binary datasets, skipped/missing features, and sample missing reasons.
2. **Dataset Dimensions & Cross-Sectional Breadth**: Segment-by-segment table showing Rows, Date Span, Trading Days, Active Tickers, and Daily Breadth (Min / Avg / Max).
3. **Model Training & Tree Topology**: Boosting rounds completed, feature count, tree max depth, and subsample/colsample parameters.
4. **Out-of-Sample Test Metrics**: Mean IC (Pearson), Rank IC (Spearman), Daily ICIR, Annualized ICIR ($\\times \\sqrt{252}$), and evaluated cross-sections.
5. **Top Factor Attribution**: Ranked factors with importance gain, mathematical formula, and economic logic.
6. **Serialized Production Artifacts**: Absolute file paths for model .pkl, booster .txt, metadata .json, and exported .parquet/.csv scores.

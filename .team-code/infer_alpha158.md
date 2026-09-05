# Technical Specification: `infer_alpha158.py`

## 1. Overview & Purpose
`infer_alpha158.py` provides high-speed, sub-millisecond inference and cross-sectional percentile ranking using the production **LightGBM Alpha158** model trained on the US Russell 1000 universe.

It is designed to serve:
1. **The Profitable Stock Trader**: Immediate pre-market retrieval of a stock's cross-sectional ranking (0–100th percentile) and conviction level without retrain lag.
2. **The Institutional Hedge Fund Manager**: Quantifiable expected forward excess return attribution with top factor drivers (e.g. `ROC20`, `MA60`, `STD20`).
3. **Automated Trading Engines**: Standardized integration endpoint for `stock_analysis_engine.py` and `stock_analysis_data.py`.

## 2. API Surface: `Alpha158Scorer`

```python
from scripts.infer_alpha158 import Alpha158Scorer

scorer = Alpha158Scorer()
result = scorer.get_score("MSFT")
print(result["alpha158_score"])     # Raw predictive score (+0.0245)
print(result["percentile"])         # Russell 1000 percentile (88.5%)
print(result["conviction_badge"])   # "🟢 STRONG LONG (TOP QUINTILE)"
```

## 3. Output Schema

```json
{
    "symbol": "MSFT",
    "as_of_date": "2026-09-04",
    "alpha158_score": 0.0245,
    "predicted_5d_excess_return": 0.0548,
    "percentile": 88.5,
    "rank": 115,
    "universe_size": 1000,
    "conviction": "STRONG BULLISH",
    "conviction_badge": "🟢 STRONG LONG (TOP QUINTILE)",
    "top_factors": [
        {"factor": "ROC20", "gain": 124.5, "impact": "Positive"},
        {"factor": "MA60", "gain": 98.2, "impact": "Positive"}
    ],
    "ic_metrics": {
        "mean_ic": 0.052,
        "rank_ic": 0.048,
        "icir": 0.68
    },
    "model_status": "TRAINED_PRODUCTION",
    "provenance": "LIGHTGBM_ALPHA158_RUSSELL1000"
}
```

## 4. CLI Usage Examples

```bash
# Query MSFT:
python scripts/infer_alpha158.py MSFT

# Query with specific as-of date:
python scripts/infer_alpha158.py NVDA --date 2026-09-01
```


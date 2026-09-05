# audit_dataset_segments Specification

## Purpose
Inspects Qlib dataset segments (	rain, alid, 	est) to extract exact observation counts, date spans, active contributing tickers, and cross-sectional breadth statistics with minimal memory consumption.

## Inputs
- dataset: Any: Instantiated Qlib DatasetH or loaded dataset from run recorder.
- segments: Dict[str, Any]: Segments dictionary mapping names to date ranges.

## Memory Strategy
Uses dataset.prepare(seg_name, col_set=["label"], data_key=DataHandlerLP.DK_L) (with DK_I fallback). Loading only the 1-column label projection preserves the entire (datetime, instrument) MultiIndex while consuming < 10 MB RAM (vs > 1.5 GB for 158 features).

## Returns
Dict[str, Any] mapping segment names to:
- ows: int: Total observation rows in segment.
- start_date: str: Earliest date in segment.
- nd_date: str: Latest date in segment.
- 	rading_days: int: Number of distinct trading dates.
- ctive_ticker_count: int: Distinct instruments contributing observations.
- ctive_tickers: List[str]: Sorted list of contributing symbols.
- daily_breadth_min: int: Minimum stocks observed on any single day.
- daily_breadth_max: int: Maximum stocks observed on any single day.
- daily_breadth_mean: float: Average daily cross-sectional stock count.
- daily_breadth_median: float: Median daily cross-sectional stock count.

# audit_universe_and_features Specification

## Purpose
Performs an exhaustive pre-flight verification of the target instrument universe against physical Qlib binary feature storage on disk before initiating model training.

## Inputs
- qlib_dir: Path: Path to Qlib data directory containing instruments/ and eatures/.
- market: str: Market universe name (e.g. ussell1000).
- start_date: str: Dataset start boundary (e.g. 2020-01-01).
- equired_features: Tuple[str, ...]: Core binary feature files (default: close.day.bin, open.day.bin, high.day.bin, low.day.bin, olume.day.bin).

## Returns
Dict[str, Any] containing:
- 	argeted_total: int: Total number of instruments defined in universe file.
- alid_count: int: Number of tickers with valid, verified binary files > 4 bytes.
- missing_count: int: Number of tickers missing the instrument directory under eatures/.
- corrupt_count: int: Number of tickers with missing or <=4 byte binary files.
- delisted_count: int: Number of tickers expired prior to dataset start date.
- 	argeted_tickers: List[str]
- alid_tickers: List[str]
- missing_tickers: List[str]
- corrupt_tickers: List[str]
- delisted_tickers: List[str]

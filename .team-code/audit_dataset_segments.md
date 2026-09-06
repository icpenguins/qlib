# audit_dataset_segments Specification

## Purpose
Inspects Qlib dataset segments (train, valid, test) to extract exact observation
counts, date spans, active contributing tickers, and cross-sectional breadth
statistics with minimal memory consumption, for the "2. DATASET DIMENSIONS &
CROSS-SECTIONAL BREADTH" section of the institutional summary banner.

## Revision History
- **2026-09-05 (this revision)**: Rewritten to take `handler_config` instead of a
  live `dataset` object. See **Root Cause of Prior N/A Output** below.
- **2026-09-05 (prior)**: Original implementation took `dataset: Any` (the object
  returned by `recorder.load_object("dataset")`) and called `dataset.prepare(...)`.
  Every segment printed `N/A` in production — see Root Cause below.

## Root Cause of Prior N/A Output
`qlib/model/trainer.py` calls `dataset.config(dump_all=False, recursive=True)`
before `R.save_objects(dataset=dataset)`. Per `Serializable.__getstate__`
(`qlib/utils/serial.py`), any attribute whose name starts with `_` — including the
handler's `_data`/`_infer`/`_learn` frames — is dropped from the pickle when
`dump_all` is false. The reloaded `dataset` object therefore has no underlying
data: `dataset.prepare(...)` raises `AttributeError` inside
`DataHandlerLP._get_df_by_key`, which the prior implementation's bare
`except Exception: continue` swallowed silently, leaving `rows == 0` for every
segment and driving the `N/A` fallback in the summary table.

The fix does not attempt to resurrect the discarded dataset (that would require
rebuilding the full handler and recomputing all 158 Alpha158 features, which is
also asymptotically wasteful just to report row counts). Instead it audits
directly from the run's `task.dataset.kwargs` config, before/independently of the
recorder round-trip.

## Signature
```python
def audit_dataset_segments(handler_config: Dict[str, Any], segments: Dict[str, Any]) -> Dict[str, Any]
```

## Inputs
- `handler_config: Dict[str, Any]`: The `task.dataset.kwargs.handler` block from
  the workflow YAML (e.g. `{"class": "Alpha158", "module_path": "qlib.contrib.data.handler", "kwargs": {...}}`).
- `segments: Dict[str, Any]`: The `task.dataset.kwargs.segments` mapping of
  segment name -> `[start, end]` date range (or a `slice`).

## Method
1. `_open_label_only_loader(handler_config)` instantiates the configured handler
   class with `init_data=False` — this resolves `instruments`/`start_time`/
   `end_time` and constructs the handler's real `data_loader` (inheriting `freq`,
   `filter_pipe`, `inst_processors`, and the label expression from config) without
   materializing a single row of data.
2. The label group is loaded directly via
   `loader.load_group_df(instruments, label_exprs, label_names, start_time, end_time, gp_name="label")`
   — one float column per label expression instead of the full 158-feature Alpha158
   matrix. Index is normalized to `(datetime, instrument)` via
   `convert_index_format` regardless of the loader's `swap_level` setting.
3. `DropnaLabel` (the default Alpha158 learn-processor) is mirrored via
   `.dropna(subset=label_cols)` so the reported `rows` matches the DK_L
   (learn-side) view the model is actually fit on, not the raw pre-drop count.
4. Each segment is sliced from the single loaded label frame via `_slice_segment`,
   using the same inclusive-both-ends semantics as Qlib's `fetch_df_by_index`.
5. `_empty_segment_stats(error=None)` returns a fully-keyed zero/`N/A` stats block
   — used both when a segment has 0 rows after dropna and when the loader itself
   cannot be constructed (e.g. unsupported handler/loader type) — so the summary
   table's `if s and s.get("rows", 0) > 0` branch always finds every key it reads.

## Verification
Equivalence-tested against a real `DatasetH.prepare(seg, col_set="label",
data_key=DataHandlerLP.DK_L)` built from a fully materialized Alpha158 handler
(5 tickers, 6-month window): rows, date span, trading days, active ticker count,
and daily breadth min/max/mean matched exactly on both segments tested. Against
the production Russell 1000 config (2020-01-02 -> 2026-09-02, ~1.49M rows across
3 segments), the label-only audit completes in ~6s.

## Returns
`Dict[str, Any]` mapping segment names to:
- `rows: int`: Observation rows in segment after label-NaN dropna (matches DK_L).
- `rows_before_label_dropna: int`: Row count before dropping label-NaN rows.
- `label_nan_dropped: int`: `rows_before_label_dropna - rows`.
- `start_date: str`: Earliest date in segment (`"N/A"` if empty).
- `end_date: str`: Latest date in segment (`"N/A"` if empty).
- `trading_days: int`: Number of distinct trading dates.
- `active_ticker_count: int`: Distinct instruments contributing observations.
- `active_tickers: List[str]`: Sorted list of contributing symbols.
- `daily_breadth_min: int`: Minimum stocks observed on any single day.
- `daily_breadth_max: int`: Maximum stocks observed on any single day.
- `daily_breadth_mean: float`: Average daily cross-sectional stock count.
- `daily_breadth_median: float`: Median daily cross-sectional stock count.
- `error: str` (optional): Present only when the segment or loader could not be
  audited; holds `f"{type(e).__name__}: {e}"`.

## Failure Handling
All exceptions are logged via `logger.warning` with the exception type and
message — no bare `except: continue`/`except: pass`. A loader construction
failure (bad handler config, missing `label` group) returns `_empty_segment_stats`
for every requested segment rather than partially-populated results.

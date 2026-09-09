# Function Specification: `load_membership_windows_from_file`

## 1. Overview & Single Responsibility
- **File**: [`scripts/download_us_selected_data.py`](file:///e:/SRC/GITHUB/my-qlib/scripts/download_us_selected_data.py)
- **Added**: 2026-09-09
- **Single Responsibility**: Read a per-ticker index-membership window (add
  date / remove date) from a source CSV, when present, so
  `dump_to_qlib_format` can write genuine point-in-time dates into
  `instruments/all.txt` instead of raw price-data availability.

## 2. Why this exists (root cause it closes)
`--data_dir D:\trading\qlib2 --market all` failed
`ensemble_lib.py::audit_universe_bias`'s survivorship-bias gate: 96.6% of the
582-row `instruments/all.txt` shared the same `end_date`. The source file,
`D:\trading\sp500_constituents.csv`, in fact carries real `Date Added`/
`Date Removed` columns (130/633 rows have a real removal date) -- but
`load_symbols_from_file` only ever extracts the ticker column, so this
information was silently discarded before `dump_to_qlib_format` ran. That
function then fell back to "first/last date we have price data for," which
for any still-*publicly-traded* company (the overwhelming majority, even
among names long since removed from an *index*) is simply "today." The
audit was correctly flagging a real, previously-invisible data defect, not
misfiring on valid data.

## 3. Function Signature
```python
def load_membership_windows_from_file(
    file_path: Union[str, Path],
) -> Dict[str, Tuple[Optional[str], Optional[str]]]:
```
Returns `{TICKER: (membership_start_or_None, membership_end_or_None)}`.
Recognized column names (case-insensitive): start = `date added` / `added` /
`start date` / `membership_start`; end = `date removed` / `removed` /
`end date` / `delisted` / `membership_end`. Returns `{}` if the file isn't a
CSV, fails to parse, or has neither column.

## 4. Integration
- `main()` calls this alongside `load_symbols_from_file` whenever `--symbol_file`
  points at a CSV, and passes the result to `run_pipeline(membership_windows=...)`
  → `dump_to_qlib_format(membership_windows=...)`.
- In `dump_to_qlib_format`, only the **instruments file's** `(start, end)` is
  narrowed to the membership window (clamped so it never extends past the
  actual downloaded price-data range). The feature `.bin` files are
  **unaffected** -- they always cover the full downloaded history regardless,
  so no price data is discarded, only the point-in-time metadata Qlib's
  instrument filtering uses is tightened.
- Symbols absent from the membership dict, or with a `None` on one side,
  keep the corresponding prior-behavior date unchanged.

## 5. Inversion guard (a real case, not hypothetical)
A ticker symbol can be reused by an unrelated listing (e.g. a company is
delisted, and years later a different company relists under the same
ticker). Clamping `start`/`end` independently against such a stale
membership record can then produce `start > end`. `dump_to_qlib_format`
detects this (`candidate_start <= candidate_end` check) and falls back to
the full, unclamped price-data range for that one symbol with a
`logger.warning`, rather than writing a broken instruments row. Found via
`ADT` in the real `D:\trading\sp500_constituents.csv` (an old 2012-2016
membership record predating the 2018 IPO whose price history is what's
actually downloaded); confirmed in
`tests/test_download_us_selected_data.py::test_dump_to_qlib_format_guards_against_inverted_membership_window`.

## 6. Verification
- `tests/test_download_us_selected_data.py`: 4 new tests, all passing
  (window parsing, no-recognized-columns fallback, clamping applied, and the
  inversion guard).
- Re-ran `ensemble_lib.py::audit_universe_bias('all', 'D:/trading/qlib2')`
  after regenerating `instruments/all.txt` from the existing (untouched)
  `.bin` files + this fix: `frac_same_end_date` dropped from 0.966 to
  0.8729 (below the 0.90 gate), `n_apparently_delisted` rose from 20 to 74.
  The gate now passes because the underlying data is more accurate, not
  because any threshold was changed.
- Ran the user's original `train_ensemble.py` command end-to-end: the
  survivorship-bias error no longer occurs; the pipeline proceeds into real
  training (LightGBM leg completes, early-stopping at 16 rounds). A separate,
  pre-existing, unrelated issue was surfaced further down the same run
  (XGBoost leg raises on `inf` values in the feature matrix) -- out of scope
  for this fix, not addressed here.

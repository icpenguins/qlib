# Implementation Plan: Generalize `russell1000_factor_verdict_screen.py` to Any Instrument List

**Date**: 2026-09-08
**Author**: team-code (Architect -> PM -> Approved Plan)
**Status**: Plan -- written before any code, per `.team-code/requirements.md` Part 2

---

## Priority-0 Acknowledgement (requirements.md, Priority Requirement -1)

Scoped for the two Priority-0 end-users defined in `requirements.md`:

- **The Profitable Stock Trader** -- multi-million capital at risk; needs the
  screen's per-ticker verdicts, prices, and dates to stay exactly as
  trustworthy as they are today. A renamed CLI flag or a silently different
  universe would be exactly the kind of "sterile academic vacuum" surprise
  this trader already distrusts.
- **Institutional Hedge Fund Manager** -- needs the tool to run over
  fund-specific candidate lists (a screened sub-universe, a sector basket),
  not just the fixed Russell 1000 file, without spinning up a new script.

**This is a refactor, not a signal-methodology change.** The binding
constraint: for any ticker already covered today (i.e. present in the
Russell 1000 default universe and the Alpha158 scores file), every number the
tool emits for that ticker -- score, rank, percentile, verdict, entry
corridor, best-buy date, RSI, regime, integrity flags -- must be byte-for-byte
identical before and after this change when run with no new flags. Only the
universe-selection mechanics and output-file naming change.

---

## 1. Problem Statement

`scripts/russell1000_factor_verdict_screen.py` hardcodes "Russell 1000" in its
filename, docstring, CLI description, default universe file
(`data/instruments/russell1000.txt`), HTML title/header, and output report
filename prefix. It can only ever screen the fixed 908-name Russell 1000 list.
Make it generic over any user-supplied instrument list (CSV, JSON, or a
CLI-supplied comma list) while defaulting to the existing Russell 1000 file
when no new flag is given, so every existing caller keeps working unchanged.

## 2. Non-Goals

- Not changing `analyse_symbol`, `load_local_ohlcv`, `detect_market_regime`,
  `predict_future_buy_timing`, or `classify_executive_verdict` -- the
  per-ticker analysis pipeline and its local-CSV price-source workaround are
  untouched.
- Not changing the Alpha158 scores source (`output/scores/alpha158_russell1000_
  latest.parquet`). That parquet is intrinsically the output of a Russell-1000
  -trained LightGBM model; tickers outside it will (as today) be recorded in
  `skipped` with "no Alpha158 score on the latest scoring date". Making the
  scoring pipeline generic is out of scope.
- Not renaming `scripts/get_russell1000_symbols.py`, `scripts/
  train_alpha158_lightgbm.py`, `scripts/infer_alpha158.py`, or any of their
  Russell-1000-specific artifact filenames (`alpha158_russell1000_latest.*`)
  -- those are correctly named for what they do and are out of scope.
- Not reintroducing `qlib.data.D.features` -- the local `source/*.csv` read
  path is preserved verbatim (see Price Source Warning, unchanged).

## 3. Repo Stack Found

Python 3.11, argparse-based CLI scripts, pandas/numpy, pytest test suite
(`tests/*.py`, none of which currently import this script by name -- verified
by repo-wide grep). No new dependency needed; CSV/JSON parsing uses stdlib
`json` (already imported) and `pandas.read_csv` (already imported).

## 4. Proposed Architecture & Modules

| Module / File | Responsibility | Owner |
| :--- | :--- | :--- |
| `scripts/factor_verdict_screen.py` (renamed from `russell1000_factor_verdict_screen.py`) | Generic universe resolution (CLI list / CSV / JSON / default file) + unchanged per-ticker pipeline + renamed outputs | Principal Dev |
| `scripts/factor_verdict_screen.md` (renamed from `russell1000_factor_verdict_screen.md`) | Updated spec: new CLI, precedence rule, universe-label scheme | Senior Dev |
| `scripts/verdict_taxonomy.md` | Update one consumer-reference line to the new filename | Senior Dev |
| `.team-code/20260905-russell1000_factor_verdict_screen-implementation_plan.md` / `-walkthrough.md` | Add a pointer note at the top; historical content otherwise untouched | Senior Dev |
| Smoke tests (manual, not a new pytest file -- no existing test targets this script) | Explicit-list run + default-Russell-1000 run | QA Tester |

## 5. Public API (new / changed)

```python
def load_universe(universe_file: Path) -> List[str]: ...          # unchanged behavior
def parse_tickers_arg(raw: str) -> List[str]: ...                  # NEW: comma list -> symbols
def load_tickers_csv(path: Path) -> List[str]: ...                 # NEW
def load_tickers_json(path: Path) -> List[str]: ...                # NEW
def resolve_ticker_universe(args: argparse.Namespace) -> Tuple[List[str], str, str]:
    """Returns (symbols, universe_label[filename-safe], universe_display[human])."""

def run_screen(
    symbols: List[str],                     # CHANGED: caller resolves symbols now
    market_data_root: Path = DEFAULT_MARKET_DATA_ROOT,
    scores_file: Path = DEFAULT_SCORES_FILE,
    limit: Optional[int] = None,
) -> Dict[str, Any]: ...

def build_screen_html(payload: Dict[str, Any]) -> str: ...         # unchanged signature; payload gains universe_display
```

`analyse_symbol`, `ScreenRow`, `SCREEN_LIMITATIONS`, `UNREACHABLE_VERDICTS`
are unchanged (no "russell1000" identifiers exist in them today).

## 6. CLI Design

New flags (dash-style, matching this file's own existing precedent --
`--market-data-root`, `--report-dir` -- and the user-suggested examples
`--tickers-csv` / `--tickers-json`; see Section 9 for why this overrides the
underscore-style seen in `visualize_stock_analysis.py` / `train_alpha158_
lightgbm.py`):

- `--tickers TICK1,TICK2,...` -- comma-separated list directly on the CLI.
- `--tickers-csv PATH` -- CSV with a `symbol`/`ticker` column (case-insensitive),
  or a single column (header optional -- a lone header cell that itself looks
  like a ticker is treated as data, not discarded).
- `--tickers-json PATH` -- bare JSON array of strings, or an object with a
  `tickers` or `symbols` key.
- `--universe PATH` -- **existing flag, behavior unchanged**: tab-separated
  qlib instruments file (`SYMBOL start end`), defaults to `data/instruments/
  russell1000.txt`.
- `--universe-name LABEL` -- optional override for the human display name and
  filename label; if omitted, a label is derived from the source used.

**Precedence (documented, not merely first-wins-silently):**
`--tickers` > `--tickers-csv` > `--tickers-json` > `--universe` (file,
defaulting to the Russell 1000 file). If more than one of `--tickers`,
`--tickers-csv`, `--tickers-json` is passed, the higher-precedence one is used
and a warning names which flags were ignored. Passing none of the three new
flags reproduces today's exact default behavior end-to-end.

**Zero-ticker error:** after resolution (and after de-dup/normalize), an empty
symbol list raises a `ValueError` naming the source that was used, caught in
`main()` and printed as a clear CLI error (exit code 2) -- never a silent
empty report.

## 7. Output Naming

Old: `reports/russell1000_factor_verdict_screen_{alpha_as_of}.html` / `.json`
New: `reports/factor_verdict_screen_{universe_label}_{alpha_as_of}.html` / `.json`

`universe_label` is a filename-safe slug: `"russell1000"` when the default
file path is used unchanged, the file stem for `--universe`/`--tickers-csv`/
`--tickers-json`, `"custom"` for `--tickers`, or the user's `--universe-name`
(slugified) when given. This is not byte-identical to the old filename (the
script itself was renamed, so full preservation is impossible), but it stays
immediately recognizable and greppable -- documented as a deliberate,
disclosed naming change, not a silent regression.

## 8. Technical Risk Matrix

- **Silent key-mismatch regression (the 2026-09-05/06 bug class):** Zero
  `ScreenRow` field names or dict keys change. `run_screen`'s payload keeps
  every existing key and adds one (`universe_display`). Verified by re-reading
  every producer/consumer pair touched before editing.
- **Behavior drift for existing callers:** Guarded by the smoke test that runs
  the renamed script with **no new flags** and confirms the Russell 1000
  default path still resolves 908 symbols from the same file via the same
  `load_universe` function, unchanged.
- **CSV/JSON ambiguity (headerless single-column CSV):** Heuristic documented
  in Section 6 and in the spec; a CSV whose sole column cannot be resolved
  raises a clear `ValueError` rather than guessing wrong silently.
- **Price-source workaround regression:** `load_local_ohlcv` is not touched;
  diffed line-for-line against the original before commit.

## 9. Program Manager Review & Approved Plan

| PM Recommendation | Decision (Keep / Drop) | Rationale & Industry Practice |
| :--- | :---: | :--- |
| Use `argparse` mutually-exclusive group (`add_mutually_exclusive_group`) for the three ticker-source flags instead of manual precedence | Drop | argparse's mutually-exclusive group would *reject* combined flags outright, but the task spec explicitly asks for a documented precedence order as an acceptable alternative to strict exclusivity ("or that the precedence order is clearly defined and documented if more than one is passed"). Manual precedence + a logged warning is more forgiving for scripted callers that pass a default `--tickers-csv` alongside an occasional override, and matches common CLI practice (e.g. `git`'s flag-precedence conventions) without losing clarity, since the precedence is fully documented in `--help` and the spec. |
| Use stdlib `csv` module instead of `pandas.read_csv` for `--tickers-csv` | Drop | pandas is already an unconditional import in this file and throughout the repo; reusing it avoids a second CSV-parsing code path and gets column-name matching (case-insensitive `symbol`/`ticker`) almost for free via `DataFrame.columns`. |
| Adopt underscore-style flags (`--tickers_csv`) to match `visualize_stock_analysis.py` / `train_alpha158_lightgbm.py` | Drop | Named industry practice considered: repo-internal consistency. This file already established dash-style multi-word flags (`--market-data-root`, `--report-dir`) before this change, and the task's own suggested examples (`--tickers-csv`, `--tickers-json`) use dashes. Changing this file's own existing flags to underscores would be a bigger, riskier diff for no behavioral gain and would break within-file consistency; the two-script convention is followed for single-word flags where there is no ambiguity. Tradeoff documented here per the Plan-Change Rule. |
| Validate CSV/JSON ticker strings against a known-symbol regex (e.g. reject anything with spaces or lowercase) | Keep | Reusing the existing `load_universe` normalization contract (`.strip().upper()`, dedup preserving order) for every new loader is a one-line reuse of an established, already-audited pattern rather than a new bespoke validator -- applied via one shared `_normalize_symbols` helper. |
| Add a new pytest file for the ticker-resolution functions | Keep | QA Tester will add unit tests for `parse_tickers_arg`, `load_tickers_csv`, `load_tickers_json`, and `resolve_ticker_universe`'s precedence/empty-universe error, since none exist today and the task requires running "any tests that reference this script or its functions by name." |

## 10. Developer Briefs

- **Principal Dev Brief:** Rename the script file (git mv), refactor `main()`
  to resolve the universe via the new precedence chain before calling
  `run_screen`, change `run_screen`'s signature to take `symbols: List[str]`
  directly, implement `parse_tickers_arg` / `load_tickers_csv` /
  `load_tickers_json` / `resolve_ticker_universe` with the shared
  `_normalize_symbols` helper, wire the empty-universe `ValueError` into
  `main()`'s error handling (exit code 2), and update the output filename
  scheme. Do not touch `analyse_symbol`, `load_local_ohlcv`,
  `load_latest_alpha_scores`, or any `ScreenRow` field.
- **Senior Dev Brief:** Rewrite module docstring, CLI `description=`/`help=`
  strings, HTML `<title>`/`<h1>`/footer text to be generic and to surface
  `universe_display`; rename and rewrite `scripts/factor_verdict_screen.md`
  (new CLI section, precedence table, output-naming section, CSV/JSON format
  docs); update the one consumer-reference line in `scripts/
  verdict_taxonomy.md`; add pointer notes atop the two historical
  `.team-code/20260905-russell1000_factor_verdict_screen-*.md` docs (content
  otherwise untouched); write unit tests for the new loader/precedence
  functions; run the full `pytest tests/ -q` suite and the two required smoke
  tests (explicit 3-5 ticker list; default no-flags Russell-1000 run),
  capturing real output.

## 11. Verification Plan

1. `python -m pytest tests/ -q` (full suite; anti-hallucination rule -- real
   output captured, blockers reported verbatim if any).
2. New unit tests for the ticker-resolution functions.
3. Smoke test A: `python scripts/factor_verdict_screen.py --tickers AAPL,MSFT,NVDA,GOOGL,AMZN --limit 5` (or without `--limit`, since an explicit list is already small) -- confirm an HTML+JSON report is produced with `universe_label` reflecting the custom list.
4. Smoke test B: `python scripts/factor_verdict_screen.py --limit 25` (no universe flags) -- confirm the default Russell 1000 file path (908-name file, `load_universe`) still resolves and the run completes end-to-end, and spot-check that per-ticker numbers for a shared ticker match a pre-refactor run byte-for-byte.

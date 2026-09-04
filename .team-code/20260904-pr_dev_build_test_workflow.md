# GitHub Actions PR Build & Test Workflow Specification (`dev` Branch)

## 1. Overview & Objective
This workflow provides automated continuous integration (CI) for pull requests targeting the `dev` branch of the repository. It guarantees that all code submitted to `dev` builds cleanly across platforms, passes fast syntax checks, compiles Cython C++ extensions in-place, and succeeds across our core institutional quantitative test suite—without triggering releases or publishing artifacts to registries.

## 2. End-User Alignment
Per `.team-code/requirements.md`:
- **The Profitable Stock Trader**: Ensures that production alpha factors (AVWAP, GEX, PEAD, Bayesian Regime Filters) and data downloaders do not suffer regression or execution failures due to broken builds or syntax defects.
- **The Institutional Hedge Fund Manager**: Mandates deterministic cross-platform validation, wheel packaging integrity checks, and 100% test passing rates across institutional unit test suites.

## 3. Workflow Architecture (`.github/workflows/pr_dev_build_test.yml`)
### Triggers & Concurrency
- **Event**: `pull_request` targeting branch `dev`, plus manual triggering via `workflow_dispatch`.
- **Concurrency**: `group: pr-dev-${{ github.event.pull_request.number || github.ref }}` with `cancel-in-progress: true` to avoid redundant runner usage on rapid pushes.

### Strict Non-Release Security Posture
- Explicitly restricted permissions:
  ```yaml
  permissions:
    contents: read
    pull-requests: read
  ```
- No release tools (`release-please`, `semantic-release`, `twine upload`, or GitHub release steps).
- Wheel artifacts built during the workflow are uploaded to GitHub Actions ephemeral job artifacts for 3 days strictly for review/debugging.

### Two-Stage Pipeline
1. **`lint-and-audit` (Fast Fail)**:
   - Python 3.11 environment on `ubuntu-latest`.
   - Runs `flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics --exclude=.venv,build,dist`. Fails immediately if syntax errors (`E999`), undefined names (`F821`), or fatal AST errors exist.
   - Runs non-fatal style audit (`--exit-zero`) to log warnings without blocking PRs.
2. **`build-and-test`**:
   - Matrix across operating systems (`ubuntu-latest`, `windows-latest`) on Python 3.11.
   - In-place Cython C++ extension compilation for high-performance rolling and expanding window operators (`rolling.pyx`, `expanding.pyx`).
   - Editable install (`pip install -e .`).
   - Package wheel packaging test (`python -m build --wheel`).
   - Execution of `scripts/run_all_tests.py -v` (core institutional suite of 52 unit tests).
   - Fast pytest run excluding slow/benchmark suites.

## 4. Syntax & Upstream CI Bug Fixes
During initial CI execution, three legacy syntax defects were identified and resolved:
1. `examples/rl_order_execution/scripts/merge_orders.py`:
   - **Defect**: `F821 undefined name 'pickle'` on line 17.
   - **Fix**: Added `import pickle` to module header.
2. `scripts/data_collector/base.py`:
   - **Defect**: `E999 IndentationError` caused by an orphaned, un-indented duplicate `def __init__` signature right before the actual method definition.
   - **Fix**: Removed the orphaned signature line.
3. `scripts/data_collector/yahoo/collector.py`:
   - **Defect**: `E999 IndentationError` caused by an orphaned, un-indented duplicate `def __init__` signature right before the actual method definition.
   - **Fix**: Removed the orphaned signature line.
4. `.github/workflows/lint_title.yml`:
   - **Defect**: `SyntaxError: Invalid regular expression flags` caused by Node.js 16 lacking support for the ECMAScript RegExp `v` flag used in modern `@commitlint` dependencies (`string-width@7+`).
   - **Fix**: Upgraded `node-version: '16'` to `'20'`.
   - **PR Title Format Requirement**: Enforced conventional commit title formatting (e.g., `feat(events): Event Risk and PEAD model` or `feat: Event Risk and PEAD model`).

## 5. Verification
- `flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics --exclude=.venv,build,dist` returns `0` fatal errors repository-wide.
- All 52 institutional unit tests passing via `scripts/run_all_tests.py`.



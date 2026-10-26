# CI pipeline

The GitHub Actions workflow is defined in
[`../.github/workflows/ci.yml`](../.github/workflows/ci.yml). It runs for pushes
to `main` or `develop` and for pull requests targeting either branch. The
workflow uses Python 3.11 and read-only repository permissions.

## Jobs and order

| Job | Depends on | Purpose |
| --- | --- | --- |
| `build-and-lint` | — | Installs dependencies, compiles `src` and `tests`, and runs Flake8. |
| `test-suite` | `build-and-lint` | Generates Iris data, runs collection, preprocessing, feature engineering and validation, then executes the full pytest suite with coverage and JUnit XML reports. |
| `model-quality-gate` | `test-suite` | Runs the dedicated model-quality tests against the minimum accuracy threshold. |

The dependency chain is `build-and-lint` → `test-suite` →
`model-quality-gate`. A failed job prevents its dependent jobs from running.
The full test suite already includes the model-quality tests; the final job
repeats those tests as a separately visible CI check.

The workflow generates `data/raw/iris_v1.csv` from scikit-learn rather than
requiring a DVC remote or credentials. The test job uploads
`test-results.xml` and `coverage.xml` as the `test-and-coverage-reports`
artifact, including when an earlier test step fails and one or both reports
were generated. It also publishes a test-count table, failure details, and
line-coverage percentage to the GitHub Actions run summary, even when the test
step fails.

To view the report on GitHub, open the repository's **Actions** tab, select a
workflow run, and open its **Summary** page. Download the detailed JUnit XML
and coverage XML from the **Artifacts** section of that run.

## Model quality regression check

The committed minimum accuracy threshold is `0.90` in
[`../tests/test_model_quality.py`](../tests/test_model_quality.py). To
demonstrate the quality gate, create a temporary feature branch and raise the
threshold (for example, to `0.999`), then open a pull request targeting `main`.
The model-quality assertion in `test-suite` should fail, preventing the
dependent `model-quality-gate` from running. Do not merge that demonstration
pull request; close it and discard the temporary branch/change afterward.
This is a procedure to verify in GitHub, not a result claimed by this
repository's local test run.

## Branch protection

GitHub branch protection is configured in the repository settings and cannot
be enabled by this workflow file. To enforce the CI gate, configure a ruleset
targeting `main`, require pull requests and passing status checks, and select
the three job checks: `build-and-lint`, `test-suite`, and
`model-quality-gate`. Verify the exact check labels shown by GitHub after the
first successful workflow run. A workflow file alone does not activate branch
protection.

## Local checks

Run the checks from the repository root:

```bash
python -m compileall -q src tests
python -m flake8 src/ tests/ --max-line-length=100 --extend-ignore=E203,W292 --statistics
python -m pytest tests/ -v --cov=src --cov-report=term-missing --junitxml=test-results.xml
```

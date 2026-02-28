# CA Module Test Fixtures

Test fixtures for the Community Associations (CA) module evaluation harness.

## Files

### `sample_insurance_policy.json`
Structured insurance policy data for the Marshall Wells Community Association test account.
Contains three coverage lines: property, general liability, and fidelity.

- **Account**: Marshall Wells Community Association
- **Account ID**: `test-marshall-wells-001`
- **Unit count**: 124
- **Total premium**: $59,800.00
- **Coverage lines**: property (State Farm), general liability (Travelers), fidelity (Hartford)

### `sample_compliance_report.json`
Compliance analysis report derived from the Marshall Wells policy data.
Captures Fannie Mae and FHA eligibility determinations with per-coverage-line detail.

### `gold_set.json`
16-question evaluation gold set for CA module extraction accuracy testing.

- **Passing threshold**: 70% (11/16 questions)
- **Categories**: coverage_lookup, policy_dates, premium, compliance, account_info, policy_number
- **Account**: Marshall Wells Community Association

Each question includes:
- `expected_answer` — canonical answer string
- `acceptable_variants` — alternative phrasings that are considered correct
- `expected_entities` — typed entity extractions expected in the answer
- `weight` — scoring weight (all 1.0 for this gold set)

## Usage

The gold set is consumed by `GoldSetRunner` in `gold_set_runner.py`:

```python
from ai_ready_rag.modules.community_associations.tests.gold_set_runner import GoldSetRunner

runner = GoldSetRunner()
result = runner.evaluate(my_answer_fn)
print(result.summary())
```

The `answer_fn` receives a question string and must return a string answer.

## Ship Gate

The CA module requires >= 70% accuracy on this gold set before merging to main.

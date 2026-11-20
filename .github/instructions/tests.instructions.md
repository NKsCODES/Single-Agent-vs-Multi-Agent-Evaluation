---
applyTo: "tests/**/*.py"
---
- Tests run offline on the simulator; never require Azure credentials.
- Assert on behaviour (answers, citations, ledger contents, HTTP status), not on log text.
- For Cosmos DB, use the fake client pattern in `tests/test_retrieval.py`.

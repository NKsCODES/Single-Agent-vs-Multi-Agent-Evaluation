.PHONY: install lint test bench gate serve docker
install:
	pip install -e ".[dev,eval,azure-ops]"
lint:
	ruff check src tests && mypy src
test:
	pytest -q --cov=archeval --cov-report=term-missing
bench:
	archeval bench --out reports/latest --repeats 5
gate:
	archeval gate reports/latest/summary.json
serve:
	uvicorn archeval.api.app:app --reload --port 8000
docker:
	docker compose up --build

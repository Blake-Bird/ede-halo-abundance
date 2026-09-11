PYTHON ?= python3

.PHONY: test validate lint typecheck check figures benchmark

test:
	PYTHONPATH=src $(PYTHON) -m pytest -q

validate:
	PYTHONPATH=src $(PYTHON) scripts/validate_hmf_foundations.py

lint:
	ruff check .
	ruff format --check .

typecheck:
	mypy src

check: test validate lint typecheck

figures:
	PYTHONPATH=src $(PYTHON) scripts/generate_hmf_foundations_figures.py

benchmark:
	PYTHONPATH=src $(PYTHON) benchmarks/benchmark_hmf_variance.py

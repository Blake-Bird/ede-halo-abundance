PYTHON ?= python3

.PHONY: test test-external validate-foundations validate-hmf solvers validate-linear validate lint typecheck check figures paper benchmark

test:
	PYTHONPATH=src $(PYTHON) -m pytest -q -m "not external_solver"

test-external:
	PYTHONPATH=src $(PYTHON) -m pytest -q -m external_solver

validate-foundations:
	PYTHONPATH=src $(PYTHON) scripts/validate_hmf_foundations.py

validate-hmf:
	PYTHONPATH=src $(PYTHON) scripts/validate_against_colossus.py

solvers:
	PYTHONPATH=src $(PYTHON) scripts/build_solvers.py

validate-linear:
	PYTHONPATH=src $(PYTHON) scripts/validate_linear_solvers.py

validate: validate-foundations solvers validate-hmf validate-linear

lint:
	ruff check .
	ruff format --check .

typecheck:
	mypy src

check: lint typecheck test validate-foundations

figures:
	PYTHONPATH=src $(PYTHON) scripts/generate_hmf_figures.py
	PYTHONPATH=src $(PYTHON) scripts/generate_linear_figures.py

paper:
	cd papers && pdflatex -interaction=nonstopmode ede_halo_abundance.tex && bibtex ede_halo_abundance && pdflatex -interaction=nonstopmode ede_halo_abundance.tex && pdflatex -interaction=nonstopmode ede_halo_abundance.tex

benchmark:
	PYTHONPATH=src $(PYTHON) benchmarks/benchmark_hmf_variance.py

PYTHON := python
UV := uv

# Hard cap on paid-API spend for the whole project (NFR-4).
API_BUDGET_USD := 500

.PHONY: help venv install lint fmt typecheck test demo clean full

help:
	@echo "semopt — cost-optimal query engine for LLM semantic operators"
	@echo ""
	@echo "  make venv       create the uv virtualenv (.venv)"
	@echo "  make install    install semopt + dev/mlx/api extras into .venv"
	@echo "  make lint       ruff check"
	@echo "  make fmt        black + ruff --fix"
	@echo "  make typecheck  mypy strict on src/"
	@echo "  make test       pytest with coverage"
	@echo "  make demo       run the M0 end-to-end demo (sem_filter over 100 rows)"
	@echo "  make full       run the whole eval pipeline (target: <2h on a laptop)"

venv:
	$(UV) venv

install:
	$(UV) pip install -e ".[dev,mlx,api]"

lint:
	$(UV) run ruff check src tests

fmt:
	$(UV) run black src tests
	$(UV) run ruff check --fix src tests

typecheck:
	$(UV) run mypy

test:
	$(UV) run pytest --cov=src/semopt --cov-report=term-missing

demo:
	$(UV) run python -m semopt.demo

coverage:
	$(UV) run python experiments/run_coverage.py

ablation:
	$(UV) run python experiments/run_ablations.py

pareto:
	$(UV) run python experiments/run_pareto.py

figures:
	$(UV) run python experiments/make_figures.py

dashboard:
	$(UV) run python experiments/run_dashboard.py

full:
	$(MAKE) coverage
	$(MAKE) ablation
	$(MAKE) pareto
	$(MAKE) dashboard
	$(MAKE) figures

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache **/__pycache__ *.egg-info

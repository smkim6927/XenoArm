.PHONY: setup check test smoke visual
setup:
	bash scripts/setup.sh
check:
	.venv/bin/ruff check src scripts tests
test:
	.venv/bin/pytest -q
smoke:
	.venv/bin/xenoarm run --config configs/smoke.json --output results/smoke-$$(date -u +%Y%m%dT%H%M%S)
visual:
	.venv/bin/xenoarm run --config configs/smoke.json --output results/visual-$$(date -u +%Y%m%dT%H%M%S) --progress --wandb offline --video

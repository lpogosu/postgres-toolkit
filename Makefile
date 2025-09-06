.PHONY: help install demo demo-down bench lint typecheck test test-fast pitr check clean

PYTHON ?= python
# Both majors by default: the version-portability tests are the reason the
# fixture is parameterised, and running one of them proves half of nothing.
PGTK_TEST_IMAGES ?= postgres:17-alpine,postgres:16-alpine

help:
	@echo "make demo       spin up PostgreSQL, seed the pathologies, run every command"
	@echo "make demo-down  remove the demo container and its volume"
	@echo "make bench      estimate vs pgstattuple: cost and disagreement"
	@echo "make pitr       base backup, WAL archive, restore to a point in time, verify"
	@echo "make test       pytest against $(PGTK_TEST_IMAGES)"
	@echo "make test-fast  only the tests that need no container"
	@echo "make check      lint + typecheck + test, the same set CI runs"

install:
	$(PYTHON) -m pip install -e ".[dev]"

demo:
	docker compose up -d --wait
	$(PYTHON) -m demo.run

demo-down:
	docker compose down -v

bench:
	$(PYTHON) -m demo.bench

lint:
	$(PYTHON) -m ruff check .

typecheck:
	$(PYTHON) -m mypy

test:
	PGTK_TEST_IMAGES=$(PGTK_TEST_IMAGES) $(PYTHON) -m pytest

test-fast:
	$(PYTHON) -m pytest -m "not pg"

pitr:
	bash backup/pitr.sh

check: lint typecheck test

clean:
	rm -rf .mypy_cache .ruff_cache .pytest_cache .demo-cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +

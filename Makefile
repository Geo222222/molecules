.PHONY: install test lint data train
install:
	pip install -e '.[dev]'
test:
	pytest -q
lint:
	ruff check src tests
data:
	python scripts/download_massspecgym.py
train:
	python -m lcms2smiles.train --config configs/base.yaml

.PHONY: venv test copy-data clean

VENV ?= neuralnet-env
PYTHON = $(VENV)/bin/python
# Where `copy-data` copies from; override with `make copy-data DATA_SRC=/other/path`.
DATA_SRC ?= /mnt/shared/storage03/projects/cern/lorenzetti/r1/parquet
DATA_DST ?= data/

venv:
	rm -rf $(VENV)
	python3 -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest -q

copy-data:
	mkdir -p $(DATA_DST)
	cp -r $(DATA_SRC) $(DATA_DST)

clean:
	rm -rf $(VENV) .pytest_cache *.egg-info
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +

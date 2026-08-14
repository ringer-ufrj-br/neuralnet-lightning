.PHONY: venv clean copy-data

VENV = neuralnet-env
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

venv:
	rm -rf neuralnet-env
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

copy-data:
	cp -r /mnt/shared/storage03/projects/cern/lorenzetti/r1/parquet data/

clean:
	rm -rf $(VENV)
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete

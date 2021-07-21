
PYTHON_BIN_PATH = python
PY_VERSION ?= 3.8

all: test lint pip_pkg

.PHONY: lint
lint: $(wildcard tensorflow_mri/python/ops/*.py)
	pylint --rcfile=pylintrc tensorflow_mri/python

.PHONY: test
test: $(wildcard tensorflow_mri/python/ops/*.py)
	$(PYTHON_BIN_PATH) -m unittest discover -v -p *_test.py

.PHONY: pip_pkg
pip_pkg:
	./build_pip_pkg.sh make artifacts

.PHONY: clean
clean:
	rm -rf artifacts/

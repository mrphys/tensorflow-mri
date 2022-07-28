CXX := /dt9/usr/bin/g++
PY_VERSION ?= 3.8
PYTHON := python$(PY_VERSION)

ROOT_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

SOURCES := $(wildcard tensorflow_mri/cc/kernels/*.cc) $(wildcard tensorflow_mri/cc/ops/*.cc)

TARGET := tensorflow_mri/python/ops/_mri_ops.so

TF_CFLAGS := $(shell $(PYTHON) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LDFLAGS := $(shell $(PYTHON) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

CFLAGS := -O3 -march=x86-64 -mtune=generic

CXXFLAGS := $(CFLAGS)
CXXFLAGS += $(TF_CFLAGS) -fPIC -std=c++14 -fopenmp
CXXFLAGS += -I$(ROOT_DIR)

LDFLAGS := $(TF_LDFLAGS)
LDFLAGS += -lfftw3_omp -lfftw3f_omp -lfftw3 -lfftw3f -lm
LDFLAGS += -l:libspiral_waveform.a

all: lib wheel

lib: $(TARGET)

$(TARGET): $(SOURCES)
	$(CXX) -shared $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

wheel: $(TARGET)
	./tools/build/build_pip_pkg.sh make --python $(PYTHON) artifacts

docs: $(TARGET)
	rm -rf tools/docs/_build
	rm -rf tools/docs/_templates
	rm -rf tools/docs/api_docs
	$(PYTHON) tools/docs/create_templates.py
	$(PYTHON) tools/docs/create_documents.py
	$(MAKE) -C tools/docs dirhtml PY_VERSION=$(PY_VERSION)

test: $(wildcard tensorflow_mri/python/*.py)
	$(PYTHON) -m unittest discover -v -p *_test.py

doctest: $(wildcard tensorflow_mri/python/*.py)
	$(PYTHON) tools/docs/test_docs.py

lint: $(wildcard tensorflow_mri/python/*.py)
	pylint --rcfile=pylintrc tensorflow_mri/python

api: $(wildcard tensorflow_mri/python/*.py)
	$(PYTHON) tools/build/create_api.py

clean:
	rm -rf artifacts/
	rm -rf $(TARGET)

.PHONY: all lib wheel test lint docs clean

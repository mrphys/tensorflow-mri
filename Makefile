CXX := g++
PY_VERSION ?= 3.8
PYTHON := python$(PY_VERSION)

ROOT_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

SOURCES := $(wildcard tensorflow_mri/cc/kernels/*.cc) $(wildcard tensorflow_mri/cc/ops/*.cc)

TARGET := tensorflow_mri/python/ops/_mri_ops.so

TF_CFLAGS := $(shell $(PYTHON) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LDFLAGS := $(shell $(PYTHON) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

CFLAGS := -O3 -march=x86-64 -mtune=generic

CXXFLAGS := $(CFLAGS)
CXXFLAGS += $(TF_CFLAGS) -fPIC -std=c++11
CXXFLAGS += -I$(ROOT_DIR)

LDFLAGS := $(TF_LDFLAGS)
LDFLAGS += -l:libspiral_waveform.a

all: lib wheel

lib: $(TARGET)

$(TARGET): $(SOURCES)
	$(CXX) -shared $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

wheel: $(TARGET)
	./tools/build/build_pip_pkg.sh make --python $(PYTHON) artifacts

docs: $(TARGET)
	ln -sf tensorflow_mri tfmr
	rm -rf tools/docs/_*
	$(MAKE) -C tools/docs html PY_VERSION=$(PY_VERSION)
	rm tfmr

test: $(wildcard tensorflow_mri/python/ops/*.py)
	$(PYTHON) -m unittest discover -v -p *_test.py

lint: $(wildcard tensorflow_mri/python/ops/*.py)
	pylint --rcfile=pylintrc tensorflow_mri/python

clean:
	rm -rf artifacts/
	rm -rf $(TARGET)

.PHONY: all lib wheel test lint docs clean

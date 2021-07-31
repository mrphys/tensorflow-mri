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

SWF_DIR := third_party/spiral_waveform
SWF_LIB := $(SWF_DIR)/lib/libspiral_waveform.a

all: lib wheel

lib: $(TARGET)

$(TARGET): $(SOURCES) $(SWF_LIB)
	$(CXX) -shared $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(SWF_LIB): $(SWF_DIR)
	$(MAKE) -C $(SWF_DIR)

$(SWF_DIR): thirdparty

.PHONY: wheel
wheel: $(TARGET)
	./tools/build/build_pip_pkg.sh make --python $(PYTHON) artifacts

.PHONY: docs
docs: $(TARGET)
	$(MAKE) -C tools/docs html

.PHONY: test
test: $(wildcard tensorflow_mri/python/ops/*.py)
	$(PYTHON) -m unittest discover -v -p *_test.py

.PHONY: lint
lint: $(wildcard tensorflow_mri/python/ops/*.py)
	pylint --rcfile=pylintrc tensorflow_mri/python

.PHONY: clean
clean:
	rm -rf artifacts/
	rm -rf $(SWF_DIR)
	rm -rf $(TARGET)

.PHONY: thirdparty
thirdparty:
	[ ! -d $(SWF_DIR) ] && git clone https://github.com/mrphys/spiral-waveform.git $(SWF_DIR) || true

ROOT_DIR := $(shell dirname "$(realpath $(firstword $(MAKEFILE_LIST)))")
BUILD_DIR ?= build
BUILD_TYPE ?= Debug

CXXFLAGS ?= -march=native
export CXXFLAGS
CUDAARCHS ?= native
export CUDAARCHS

THRUST_DEVICE_SYSTEM ?= CUDA

FORK ?= $(shell nproc)
FUZZ_MAX_TOTAL_TIME ?= 0
FUZZ_MAX_PRIMITIVE_COUNT ?= 0
FUZZ_BOX_WORLD ?= 0
TEST_NAME_REGEX ?= .*

VENV      := venv
PYTHON    := $(VENV)/bin/python
PIP       := $(VENV)/bin/pip
PYTEST    := $(VENV)/bin/pytest
MYPY      := $(VENV)/bin/mypy
RUFF      := $(VENV)/bin/ruff
CLANG_FMT := $(VENV)/bin/clang-format
PRECOMMIT := $(VENV)/bin/pre-commit

GIT_FIRST_COMMIT := $(shell git rev-list --max-parents=0 HEAD)
SPIRV_HEADERS := $(PWD)/external/SPIRV-Headers/include

SCREEN_SIZE ?= $(shell xdpyinfo | awk '/dimensions:/ { print $$2 }' | tr 'x' ' ')

.DEFAULT_GOAL := build

.DELETE_ON_ERROR:

.ONESHELL:
SHELL := $(shell which bash)
.SHELLFLAGS = -eu -o pipefail -c

.SECONDARY: $(VENV)/bin/activate
$(VENV)/bin/activate: pyproject.toml
	trap 'rm -rf $(VENV)' ERR
	python3 -m venv venv/
	$(PIP) install --quiet --upgrade pip setuptools
	$(PIP) install --verbose --progress-bar=on --no-build-isolation --editable .[dev]
	touch $@

.PHONY: venv
venv: $(VENV)/bin/activate

.PHONY: clean-venv
clean-venv:
	rm -rf $(VENV)

.PHONY: re-venv
re-venv: clean-venv venv

.PHONY: install-hooks
install-hooks: venv
	$(PRECOMMIT) install
	$(PRECOMMIT) install --hook-type commit-msg

.PHONY: sh shell
sh shell: venv
	. venv/bin/activate
	
	$(SHELL)

.PHONY: configure
configure: venv
	nice cmake \
	    -S $(PWD) \
	    -B $(BUILD_DIR) \
	    -DTHRUST_DEVICE_SYSTEM=$(THRUST_DEVICE_SYSTEM) \
	    -DCMAKE_VERBOSE_MAKEFILE=ON \
	    -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)

.PHONY: cmake-graphviz
cmake-graphviz: venv
	cmake \
	    --graphviz=$(BUILD_DIR)/sah_kd_tree.dot \
	    -S $(PWD) \
	    -B $(BUILD_DIR) \
	    -DTHRUST_DEVICE_SYSTEM=$(THRUST_DEVICE_SYSTEM) \
	    -DCMAKE_VERBOSE_MAKEFILE=ON \
	    -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)
	dot \
	    -Tpng \
	    -o $(BUILD_DIR)/sah_kd_tree.png \
	    $(BUILD_DIR)/sah_kd_tree.dot

.PHONY: build
build: venv configure
	nice cmake \
	    --build $(BUILD_DIR) \
	    --parallel

.PHONY: rebuild
rebuild: venv configure
	nice cmake \
	    --build $(BUILD_DIR) \
	    --parallel \
	    --clean-first

.PHONY: clean
clean: venv configure
	nice cmake \
	    --build $(BUILD_DIR) \
	    --parallel \
	    --target clean

.PHONY: docker-run
docker-run: venv
	tools/docker/archlinux/run.sh $(COMMAND)

.PHONY: fuzz
fuzz: venv configure
	nice cmake \
	    --build $(BUILD_DIR) \
	    --parallel \
	    --target fuzzer
	
	nice $(BUILD_DIR)/tools/fuzz/fuzzer \
	    -box_world=$(FUZZ_BOX_WORLD) \
	    -max_primitive_count=$(FUZZ_MAX_PRIMITIVE_COUNT) \
	    -max_total_time=$(FUZZ_MAX_TOTAL_TIME) \
	    -fork=$(FORK) \
	    -rss_limit_mb=512 \
	    -timeout=30 \
	    -report_slow_units=30 \
	    -print_final_stats=1 \
	    -print_corpus_stats=1 \
	    -print_pcs=1 \
	    -reduce_depth=1 \
	    -reduce_inputs=1 \
	    -shrink=1 \
	    -prefer_small=1 \
	    -artifact_prefix=data/fuzz/artifacts/ \
	    data/fuzz/CORPUS/ \
	    data/fuzz/artifacts/

.PHONY: fuzz-merge
fuzz-merge: venv configure
	nice cmake \
	    --build $(BUILD_DIR) \
	    --parallel \
	    --target fuzzer
	
	nice tools/fuzz/fuzzer \
	    -fork=$(FORK) \
	    -merge=1 \
	    data/fuzz/CORPUS*/ \
	    data/fuzz/artifacts/

.PHONY: plan 3d
plan 3d: venv $(CRASH_FILE)
	gnuplot \
	    -persist \
	    -c tools/plot/plot.plt \
	    $@ \
	    $(CRASH_FILE) \
	    $(SCREEN_SIZE)

.PHONY: check
check: venv
	$(RUFF) check
	$(RUFF) format --check
	MYPYPATH=$(SPIRV_HEADERS) \
	$(MYPY) \
	    --exclude-gitignore \
	    src/

.PHONY: format
format: venv
	$(RUFF) format
	$(RUFF) check --fix
	MYPYPATH=$(SPIRV_HEADERS) \
	$(MYPY) \
	    --exclude-gitignore \
	    src/
	
	git add --update
	git clang-format \
	    --binary=$(CLANG_FMT) \
	    --extensions=cpp,hpp,cu,cuh,cu.inl,js \
	    $(GIT_FIRST_COMMIT) \
	|| true
	git status

.PHONY: test-cpp
test-cpp: build
	ctest \
	    --parallel \
	    --output-on-failure \
	    --test-dir $(BUILD_DIR)/src/ \
	    -R '$(TEST_NAME_REGEX)'

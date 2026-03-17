ROOT_DIR := $(shell dirname "$(realpath $(firstword $(MAKEFILE_LIST)))")
BUILD_DIR ?= $(ROOT_DIR)/build
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
PYTHON ?= python3

SCREEN_SIZE ?= $(shell xdpyinfo | awk '/dimensions:/ { print $$2 }' | tr 'x' ' ')

.DEFAULT_GOAL := build

.ONESHELL:
SHELL = bash
.SHELLFLAGS = -eu -o pipefail -c

$(ROOT_DIR)/venv/bin/activate:
	trap 'rm -rf $(ROOT_DIR)/venv/' ERR
	$(PYTHON) -m venv $(ROOT_DIR)/venv/
	. $(ROOT_DIR)/venv/bin/activate
	pip install --requirement $(ROOT_DIR)/requirements.txt

.PHONY: venv
venv: $(ROOT_DIR)/venv/bin/activate

.PHONY: sh
sh: venv
	. $(ROOT_DIR)/venv/bin/activate
	
	$(SHELL)

.PHONY: configure
configure: venv
	. $(ROOT_DIR)/venv/bin/activate
	
	nice cmake \
	    -S $(ROOT_DIR) \
	    -B $(BUILD_DIR) \
	    -DTHRUST_DEVICE_SYSTEM=$(THRUST_DEVICE_SYSTEM) \
	    -DCMAKE_VERBOSE_MAKEFILE=ON \
	    -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)

.PHONY: cmake-graphviz
cmake-graphviz: venv
	. $(ROOT_DIR)/venv/bin/activate
	
	cmake \
	    --graphviz=$(BUILD_DIR)/sah_kd_tree.dot \
	    -S $(ROOT_DIR) \
	    -B $(BUILD_DIR) \
	    -DTHRUST_DEVICE_SYSTEM=$(THRUST_DEVICE_SYSTEM) \
	    -DCMAKE_VERBOSE_MAKEFILE=ON \
	    -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
	    $(ROOT_DIR)
	dot \
	    -Tpng \
	    -o $(BUILD_DIR)/sah_kd_tree.png \
	    $(BUILD_DIR)/sah_kd_tree.dot

.PHONY: build
build: venv configure
	. $(ROOT_DIR)/venv/bin/activate
	
	nice cmake \
	    --build $(BUILD_DIR) \
	    --parallel

.PHONY:
rebuild: venv configure
	. $(ROOT_DIR)/venv/bin/activate
	
	nice cmake \
	    --build $(BUILD_DIR) \
	    --parallel \
	    --clean-first

.PHONY: venv clean
clean: configure
	. $(ROOT_DIR)/venv/bin/activate
	
	nice cmake \
	    --build $(BUILD_DIR) \
	    --parallel \
	    --target clean

.PHONY: test
test: venv build
	. $(ROOT_DIR)/venv/bin/activate
	
	ctest \
	    --parallel \
	    --output-on-failure \
	    --test-dir $(BUILD_DIR)/src/ \
	    -R '$(TEST_NAME_REGEX)'

.PHONY: docker-run
docker-run: venv
	. $(ROOT_DIR)/venv/bin/activate
	
	$(ROOT_DIR)/tools/docker/archlinux/run.sh $(COMMAND)

.PHONY: fuzz
fuzz: configure
	nice cmake \
	    --build $(BUILD_DIR) \
	    --parallel \
	    --target fuzzer
	
	tools/fuzz/fuzzer \
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
	    -artifact_prefix=$(ROOT_DIR)/data/fuzz/artifacts/ \
	    $(ROOT_DIR)/data/fuzz/CORPUS/ \
	    $(ROOT_DIR)/data/fuzz/artifacts/

.PHONY: fuzz-merge
fuzz-merge: configure
	nice cmake \
	    --build $(BUILD_DIR) \
	    --parallel \
	    --target fuzzer
	
	tools/fuzz/fuzzer \
	    -fork=$(FORK) \
	    -merge=1 \
	    $(ROOT_DIR)/data/fuzz/CORPUS*/ \
	    $(ROOT_DIR)/data/fuzz/artifacts/

.PHONY: plan 3d
plan 3d: $(CRASH_FILE)
	gnuplot \
	    -persist \
	    -c $(ROOT_DIR)/tools/plot/plot.plt \
	    $@ \
	    $(CRASH_FILE) \
	    $(SCREEN_SIZE)

.PHONY: format
format: venv
	cd $(ROOT_DIR)
	git add --update
	git clang-format --binary=venv/bin/clang-format --extensions=cpp,hpp,cu,cuh,inl,js $(shell git rev-list --max-parents=0 HEAD) || true
	. $(ROOT_DIR)/venv/bin/activate
	black src/
	isort --profile black src/
	MYPYPATH=$(ROOT_DIR)/external/SPIRV-Headers/include mypy src/
	git status

.PHONY: pytest
pytest: venv
	. $(ROOT_DIR)/venv/bin/activate
	
	pytest $(ROOT_DIR)/src/

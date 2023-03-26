NPROC ?= $(shell nproc)
FORK ?= $(shell echo $$(( $(NPROC) / 2 )))
ROOT_DIR := $(shell dirname "$(realpath $(firstword $(MAKEFILE_LIST)))")
BUILD_DIR ?= /tmp/build-sah_kd_tree
BUILD_TYPE ?= Debug
BUILD_SHARED_LIBS ?= ON
LINKER ?= $(shell which lld)
C_COMPILER ?= $(shell which clang)
C_FLAGS ?= -march=x86-64 -fno-omit-frame-pointer -fno-optimize-sibling-calls
CXX_COMPILER ?= $(shell which clang++)
CXX_FLAGS ?= -march=x86-64 -fno-omit-frame-pointer -fno-optimize-sibling-calls
CUDA_FLAGS ?= -fopenmp-version=45 -fno-omit-frame-pointer -fno-optimize-sibling-calls 
CUDA_ARCH ?= $(shell nvcc -arch=native -Xcompiler -dM -E -x cu - </dev/null | awk '/__CUDA_ARCH__/ { print $$3 / 10 }')
THRUST_DEVICE_SYSTEM ?= CPP
FUZZ_MAX_TOTAL_TIME ?= 0
FUZZ_MAX_PRIMITIVE_COUNT ?= 0
FUZZ_BOX_WORLD ?= 0
TEST_NAME_REGEX ?= .*

# format: "800 600"
SCREEN_SIZE ?= $(shell xdpyinfo | awk '/dimensions:/ { print $$2 }' | tr 'x' ' ')

.DEFAULT_GOAL := build

.PHONY: print-cuda-arch
print-cuda-arch:
	@echo $(CUDA_ARCH)

.PHONY: configure
configure:
	@cmake -E make_directory $(BUILD_DIR)
	@nice cmake \
		-S $(ROOT_DIR) \
		-B $(BUILD_DIR) \
		-DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
		-DBUILD_SHARED_LIBS=$(BUILD_SHARED_LIBS) \
		-DCMAKE_LINKER=$(LINKER) \
		-DCMAKE_C_COMPILER=$(C_COMPILER) \
		-DCMAKE_C_FLAGS="$(C_FLAGS)" \
		-DCMAKE_CXX_COMPILER=$(CXX_COMPILER) \
		-DCMAKE_CXX_FLAGS="$(CXX_FLAGS)" \
		-DCMAKE_CUDA_HOST_COMPILER=$(CXX_COMPILER) \
		-DCMAKE_CUDA_ARCHITECTURES=$(CUDA_ARCH) \
		-DCMAKE_CUDA_FLAGS="$(CUDA_FLAGS)" \
		-DCMAKE_VERBOSE_MAKEFILE=ON \
		-DTHRUST_DEVICE_SYSTEM=$(THRUST_DEVICE_SYSTEM)

.PHONY: build
build: configure
	@nice cmake \
		--build $(BUILD_DIR) \
		--parallel $(NPROC) \
		--target all

.PHONY:
rebuild: configure
	@nice cmake \
		--build $(BUILD_DIR) \
		--parallel $(NPROC) \
		--clean-first \
		--target all

.PHONY: clean
clean: configure
	@nice cmake \
		--build $(BUILD_DIR) \
		--parallel $(NPROC) \
		--target clean

.PHONY: test
test: build
	@ctest \
		--parallel $(NPROC) \
		--output-on-failure \
		--test-dir $(BUILD_DIR)/src/ \
		-R '$(TEST_NAME_REGEX)'

.PHONY: fuzz
fuzz: configure
	@nice cmake \
		--build $(BUILD_DIR) \
		--parallel $(NPROC) \
		--target fuzzer
	@tools/fuzz/fuzzer \
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
	@nice cmake \
		--build $(BUILD_DIR) \
		--parallel $(NPROC) \
		--target fuzzer
	@tools/fuzz/fuzzer \
		-fork=$(FORK) \
		-merge=1 \
		$(ROOT_DIR)/data/fuzz/CORPUS*/ \
		$(ROOT_DIR)/data/fuzz/artifacts/

.PHONY: plan 3d
plan 3d: $(CRASH_FILE)
	@gnuplot \
		-persist \
		-c $(ROOT_DIR)/tools/plot/plot.plt \
		$@ \
		$(CRASH_FILE) \
		$(SCREEN_SIZE)

.PHONY: cmake-graphviz
cmake-graphviz:
	@cmake -E make_directory $(BUILD_DIR)
	@cd $(BUILD_DIR) && \
	cmake \
		--graphviz=sah_kd_tree.dot \
		-S $(ROOT_DIR) \
		-B $(BUILD_DIR) \
		-DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
		-DBUILD_SHARED_LIBS=$(BUILD_SHARED_LIBS) \
		-DTHRUST_DEVICE_SYSTEM=$(THRUST_DEVICE_SYSTEM) \
		$(ROOT_DIR) && \
	dot -Tpng -o sah_kd_tree.png sah_kd_tree.dot && \
	xdg-open sah_kd_tree.png

.PHONY: format
format:
	@git add $(ROOT_DIR)
	@git clang-format $(shell git rev-list --max-parents=0 @) || true

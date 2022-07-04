default: build

ROOT_DIR := $(shell dirname "$(realpath $(firstword $(MAKEFILE_LIST)))")
BUILD_DIR ?= "/tmp/build-sah_kd_tree"
BUILD_TYPE ?= Release
BUILD_SHARED_LIBS ?= ON
THRUST_DEVICE_SYSTEM ?= CPP
NPROC ?= $(shell nproc)
FUZZ_FORK ?= $(shell echo $$(( $(NPROC) / 2 )))
FUZZ_DURATION ?= 0
FUZZ_MAX_PRIMITIVE_COUNT ?= 0
FUZZ_BOX_WORLD ?= 0

# format: "800 600"
SCREEN_SIZE ?= $(shell xdpyinfo | awk '/dimensions:/ { print $$2 }' | awk -F x '{ print $$1, $$2 }')

.PHONY: cmake
cmake:
	@cmake -E make_directory "$(BUILD_DIR)"
	@nice cmake \
		-S "$(ROOT_DIR)" \
		-B "$(BUILD_DIR)" \
		-DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
		-DBUILD_SHARED_LIBS=$(BUILD_SHARED_LIBS) \
		-DCMAKE_CUDA_ARCHITECTURES=86 \
		-DCMAKE_CXX_COMPILER="$$( which clang++ )" \
		-DCMAKE_CXX_FLAGS="-march=native -fno-omit-frame-pointer -fno-optimize-sibling-calls" \
		-DCMAKE_CUDA_HOST_COMPILER="$$( which clang++ )" \
		-DCMAKE_VERBOSE_MAKEFILE=ON \
		-DCMAKE_CUDA_FLAGS="-Xcompiler -fopenmp-version=45" \
		-DTHRUST_DEVICE_SYSTEM=$(THRUST_DEVICE_SYSTEM)

.PHONY: build
build: cmake
	@nice cmake \
		--build "$(BUILD_DIR)" \
		--parallel $(NPROC) \
		--target all

.PHONY:
rebuild: cmake
	@nice cmake \
		--build "$(BUILD_DIR)" \
		--parallel $(NPROC) \
		--target all \
		--clean-first

.PHONY: clean
clean:
	@nice cmake \
		--build "$(BUILD_DIR)" \
		--parallel $(NPROC) \
		--target clean

.PHONY: fuzz
fuzz:
	@nice cmake \
		--build "$(BUILD_DIR)" \
		--parallel $(NPROC) \
		--target fuzzer
	@tools/fuzz/fuzzer \
		-max_primitive_count=$(FUZZ_MAX_PRIMITIVE_COUNT) \
		-box_world=$(FUZZ_BOX_WORLD) \
		-fork=$(FUZZ_FORK) \
		-rss_limit_mb=512 \
		-timeout=30 \
		-report_slow_units=30 \
		-max_total_time=$(FUZZ_DURATION) \
		-print_final_stats=1 \
		-print_corpus_stats=1 \
		-print_pcs=1 \
		-reduce_depth=1 \
		-reduce_inputs=1 \
		-shrink=1 \
		-prefer_small=1 \
		-artifact_prefix="$(ROOT_DIR)/data/fuzz/artifacts/" \
		"$(ROOT_DIR)/data/fuzz/CORPUS/" \
		"$(ROOT_DIR)/data/fuzz/artifacts/"

.PHONY: fuzz-merge
fuzz-merge:
	@nice cmake \
		--build "$(BUILD_DIR)" \
		--parallel $(NPROC) \
		--target fuzzer
	@tools/fuzz/fuzzer \
		-fork=$(FUZZ_FORK) \
		-merge=1 \
		"$(ROOT_DIR)/data/fuzz/CORPUS"*/ \
		"$(ROOT_DIR)/data/fuzz/artifacts/"

.PHONY: plan 3d
plan 3d: $(CRASH_FILE)
	@gnuplot \
		-persist \
		-c "$(ROOT_DIR)/tools/plot/plot.plt" \
		$@ \
		"$(CRASH_FILE)" \
		$(SCREEN_SIZE)


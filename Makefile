default: build

ROOT_DIR := $(shell dirname "$(realpath $(firstword $(MAKEFILE_LIST)))")
BUILD_DIR ?= "/tmp/build-sah_kd_tree"
NPROC ?= $(shell nproc)
FUZZ_FORK ?= $(shell echo $$(( $(NPROC) / 2 )))
FUZZ_DURATION ?= 30

# format: "800 600"
SCREEN_SIZE ?= $(shell xdpyinfo | awk '/dimensions:/ { print $$2 }' | awk -F x '{ print $$1, $$2 }')

.PHONY: cmake
cmake:
	@cmake -E make_directory "$(BUILD_DIR)"
	@nice cmake -S "$(ROOT_DIR)" -B "$(BUILD_DIR)" \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_CUDA_ARCHITECTURES=86 \
        -DCMAKE_CXX_COMPILER="$$( which clang++ )" \
		-DCMAKE_CXX_FLAGS="-march=native -fno-omit-frame-pointer -fno-optimize-sibling-calls" \
		-DCMAKE_CUDA_HOST_COMPILER="$$( which clang++ )" \
		-DCMAKE_VERBOSE_MAKEFILE=ON \
		-DCMAKE_CUDA_FLAGS="-Xcompiler -fopenmp-version=45" \
		-DTHRUST_DEVICE_SYSTEM=CPP

.PHONY: build
build: cmake
	@nice cmake --build "$(BUILD_DIR)" \
		--parallel $(NPROC) \
		--target all \
		--config release

.PHONY: rebuild
rebuild: cmake
	@nice cmake --build "$(BUILD_DIR)" \
		--parallel $(NPROC) \
		--target all \
		--clean-first \
		--config release

.PHONY: clean
clean: cmake
	@nice cmake --build "$(BUILD_DIR)" \
		--parallel $(NPROC) \
		--target clean

.PHONY: fuzz
fuzz: build
	@nice cmake --build "$(BUILD_DIR)" \
		--parallel $(NPROC) \
		--target fuzzer \
		--config release
	@tools/fuzz/fuzzer \
		-fork=$(FUZZ_FORK) \
		-use_value_profile=1 \
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

.PHONY: plan 3d
plan 3d: $(CRASH_FILE)
	@gnuplot -persist \
		-c "$(ROOT_DIR)/tools/plot/plot.plt" \
		$@ \
		"$(CRASH_FILE)" \
		$(SCREEN_SIZE)


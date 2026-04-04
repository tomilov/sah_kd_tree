#pragma once

#include <compute/format.hpp>
#include <compute/fwd.hpp>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cufile.h>

#include <compute/compute_export.h>

#define CU_CALL_CHECK(f, ...)                                                                  \
    do {                                                                                       \
        ::CUresult result{(f(__VA_ARGS__))};                                                   \
        SKT_INVARIANT(result == CUDA_SUCCESS, SKT_STRINGIZE(f(__VA_ARGS__)) " -> {}", result); \
    } while (false)

#define CUDA_CALL_CHECK(f, ...)                                                             \
    do {                                                                                    \
        cudaError error{(f(__VA_ARGS__))};                                                  \
        SKT_INVARIANT(error == cudaSuccess, SKT_STRINGIZE(f(__VA_ARGS__)) " -> {}", error); \
    } while (false)

#define CUFILE_CALL_CHECK(f, ...)                                                                                        \
    do {                                                                                                                 \
        constexpr auto abs = [](CUfileOpError opError) -> CUfileOpError                                                  \
        {                                                                                                                \
            return opError;                                                                                              \
        };                                                                                                               \
        CUfileError_t error{(f(__VA_ARGS__))};                                                                           \
        SKT_INVARIANT(!(IS_CUDA_ERR(error) && IS_CUFILE_ERR(error.err)), SKT_STRINGIZE(f(__VA_ARGS__)) " -> {}", error); \
    } while (false)

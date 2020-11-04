#pragma once

#ifndef __CUDACC_EXTENDED_LAMBDA__
#error "nvcc --extended-lambda"
#endif
#ifndef __CUDACC_RELAXED_CONSTEXPR__
#error "nvcc --expt-relaxed-constexpr"
#endif

#include <thrust/tuple.h>

#include <limits>

namespace sah_kd_tree
{
using I = int;
using U = unsigned int;
using F = float;

struct params
{
    F emptiness_factor = 0.8f;   // (0, 1]
    F traversal_cost = 2.0f;     // (0, inf)
    F intersection_cost = 1.0f;  // (0, inf)
    U max_depth = std::numeric_limits<U>::max();
};

template <typename T>
struct add_leaf_const_reference
{
    using type = const T &;  // I, U, F
};

template <typename T>
using add_leaf_const_reference_t = typename add_leaf_const_reference<T>::type;

template <>
struct add_leaf_const_reference<thrust::null_type>
{
    using type = thrust::null_type;
};

template <typename... Types>
struct add_leaf_const_reference<thrust::tuple<Types...>>
{
    using type = thrust::tuple<add_leaf_const_reference_t<Types>...>;
};

template <typename iterator>
using input_iterator_value_type = add_leaf_const_reference_t<typename iterator::value_type>;

template <typename iterator>
using output_iterator_value_type = typename iterator::value_type;

}  // namespace sah_kd_tree

#include <type_traits>

#include <thrust/system/cuda/execution_policy.h>
#include <thrust/system/cuda/pointer.h>
#include <thrust/system/cuda/vector.h>

namespace sah_kd_tree
{
using policy = decltype(thrust::cuda::par);

template <typename... Types>
using container = thrust::cuda::vector<Types...>;

template <I dimension>
struct projection
{
    policy p;
};

extern template struct projection<0>;
extern template struct projection<1>;
extern template struct projection<2>;

struct builder
{
    policy p;

    projection<0> x{p};
    projection<1> y{p};
    projection<2> z{p};
};

inline builder make_builder(const policy & p)
{
    return {p};
}

}  // namespace sah_kd_tree

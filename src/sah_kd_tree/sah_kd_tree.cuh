#pragma once

#include <type_traits>

#include <sah_kd_tree/config.hpp>

#include <thrust/system/cuda/vector.h>

namespace SahKdTree
{
template<I dimension>
struct Projection
{
    struct
    {
        thrust::cuda::vector<F> a, b, c;
    } triangle;
};

extern template struct Projection<0>;
extern template struct Projection<1>;
extern template struct Projection<2>;

struct Builder
{
    Projection<0> x;
    Projection<1> y;
    Projection<2> z;

    void operator()(const Params & /*sah*/)
    {}
};
}  // namespace SahKdTree

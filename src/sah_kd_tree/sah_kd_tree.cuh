#pragma once

#include <sah_kd_tree/config.hpp>

#include <thrust/system/cuda/vector.h>

#include <type_traits>

namespace SahKdTree
{
template<I dimension>
struct Projection
{
    using X = Projection;
    using Y = Projection<(dimension + 1) % 3>;
    using Z = Projection<(dimension + 2) % 3>;

    struct
    {
        thrust::cuda::vector<F> a, b, c;
    } triangle;

    struct
    {
        thrust::cuda::vector<F> min, max;
    } polygon;

    void calculateTriangleBbox();
};

struct Builder
{
    Projection<0> x;
    Projection<1> y;
    Projection<2> z;

    void operator()(const Params & sah);
};
}  // namespace SahKdTree

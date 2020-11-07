#pragma once

#include <sah_kd_tree/config.hpp>

#include <thrust/device_vector.h>

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
        thrust::device_vector<F> a, b, c;
    } triangle;

    struct
    {
        thrust::device_vector<F> min, max;
    } polygon;

    struct
    {
        thrust::device_vector<F> min, max;
    } node;

    struct
    {
        thrust::device_vector<U> node;
        thrust::device_vector<F> pos;
        thrust::device_vector<I> kind;
        thrust::device_vector<U> polygon;
    } event;

    void calculateTriangleBbox();
    void caluculateRootNodeBbox();
    void generateInitialEvent();
};

struct Builder
{
    Projection<0> x;
    Projection<1> y;
    Projection<2> z;

    void operator()(const Params & sah);
};
}  // namespace SahKdTree

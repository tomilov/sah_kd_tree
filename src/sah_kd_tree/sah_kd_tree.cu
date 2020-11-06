#include "sah_kd_tree.cuh"

#include <sah_kd_tree/sah_kd_tree.hpp>

#include <thrust/advance.h>
#include <thrust/distance.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/system/cuda/vector.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#ifndef __CUDACC_EXTENDED_LAMBDA__
#error "nvcc --extended-lambda"
#endif
#ifndef __CUDACC_RELAXED_CONSTEXPR__
#error "nvcc --expt-relaxed-constexpr"
#endif

namespace SahKdTree
{
template<typename Iterator>
using IteratorValueType = typename Iterator::value_type;

template struct Projection<0>;
template struct Projection<1>;
template struct Projection<2>;

template<Vertex Triangle::*V, float Vertex::*C>
struct TriangleSlice
{
    constexpr float operator()(const Triangle & t)
    {
        return t.*V.*C;
    }
};

void build(const Params & sah, thrust::cuda::pointer<const Triangle> triangleBegin, thrust::cuda::pointer<const Triangle> triangleEnd)
{
    Builder builder;
    auto triangleCount = thrust::distance(triangleBegin, triangleEnd);
    if ((true)) {
        auto ax = thrust::make_transform_iterator(triangleBegin, TriangleSlice<&Triangle::a, &Vertex::x>{});
        auto ay = thrust::make_transform_iterator(triangleBegin, TriangleSlice<&Triangle::a, &Vertex::y>{});
        auto az = thrust::make_transform_iterator(triangleBegin, TriangleSlice<&Triangle::a, &Vertex::z>{});

        auto bx = thrust::make_transform_iterator(triangleBegin, TriangleSlice<&Triangle::b, &Vertex::x>{});
        auto by = thrust::make_transform_iterator(triangleBegin, TriangleSlice<&Triangle::b, &Vertex::y>{});
        auto bz = thrust::make_transform_iterator(triangleBegin, TriangleSlice<&Triangle::b, &Vertex::z>{});

        auto cx = thrust::make_transform_iterator(triangleBegin, TriangleSlice<&Triangle::c, &Vertex::x>{});
        auto cy = thrust::make_transform_iterator(triangleBegin, TriangleSlice<&Triangle::c, &Vertex::y>{});
        auto cz = thrust::make_transform_iterator(triangleBegin, TriangleSlice<&Triangle::c, &Vertex::z>{});

        auto setTriangleProjection = [triangleCount](auto & projection, auto a, auto b, auto c) {
            auto & triangle = projection.triangle;
            triangle.a.assign(a, thrust::next(a, triangleCount));
            triangle.b.assign(b, thrust::next(b, triangleCount));
            triangle.c.assign(c, thrust::next(c, triangleCount));
        };
        setTriangleProjection(builder.x, ax, bx, cx);
        setTriangleProjection(builder.y, ay, by, cy);
        setTriangleProjection(builder.z, az, bz, cz);
    } else {
        Builder builder;
        auto resizeTriangleProjection = [triangleCount](auto & projection) {
            auto & triangle = projection.triangle;
            triangle.a.resize(size_t(triangleCount));
            triangle.b.resize(size_t(triangleCount));
            triangle.c.resize(size_t(triangleCount));
        };
        resizeTriangleProjection(builder.x);
        resizeTriangleProjection(builder.y);
        resizeTriangleProjection(builder.z);

        auto splitTriangleProjection = [](auto & projection) {
            auto & triangle = projection.triangle;
            return thrust::make_zip_iterator(thrust::make_tuple(triangle.a.begin(), triangle.b.begin(), triangle.c.begin()));
        };
        auto splittedTriangleBegin = thrust::make_zip_iterator(thrust::make_tuple(splitTriangleProjection(builder.x), splitTriangleProjection(builder.y), splitTriangleProjection(builder.z)));
        using SplittedTriangleType = IteratorValueType<decltype(splittedTriangleBegin)>;
        auto splitTriangle = [] __host__ __device__(const Triangle & t) -> SplittedTriangleType { return {{t.a.x, t.a.y, t.a.z}, {t.b.x, t.b.y, t.b.z}, {t.c.x, t.c.y, t.c.z}}; };
        thrust::transform(triangleBegin, triangleEnd, splittedTriangleBegin, splitTriangle);
    }
    return builder(sah);
}
}  // namespace SahKdTree

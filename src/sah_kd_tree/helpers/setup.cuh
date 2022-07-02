#pragma once

#include <sah_kd_tree/types.cuh>
#include <sah_kd_tree/utility.cuh>

#include <thrust/advance.h>
#include <thrust/device_vector.h>
#include <thrust/distance.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

namespace sah_kd_tree::helpers
{
struct Triangles
{
    U triangleCount = 0;
    struct Projection
    {
        thrust::device_vector<F> a, b, c;
    } x, y, z;
};

template<typename TriangleIterator>
void setTriangles(Triangles & triangles, TriangleIterator triangleBegin, TriangleIterator triangleEnd)
{
    Timer timer;

    using TriangleType = IteratorValueType<TriangleIterator>;
    triangles.triangleCount = U(thrust::distance(triangleBegin, triangleEnd));
    auto transposeProjection = [triangleCount = triangles.triangleCount](typename Triangles::Projection & projection)
    {
        projection.a.resize(std::size_t(triangleCount));
        projection.b.resize(std::size_t(triangleCount));
        projection.c.resize(std::size_t(triangleCount));
        return thrust::make_zip_iterator(thrust::make_tuple(projection.a.begin(), projection.b.begin(), projection.c.begin()));
    };
    auto transposedTriangleBegin = thrust::make_zip_iterator(thrust::make_tuple(transposeProjection(triangles.x), transposeProjection(triangles.y), transposeProjection(triangles.z)));
    using TransposedTriangleType = IteratorValueType<decltype(transposedTriangleBegin)>;
    auto transposeTriangle = [] __host__ __device__(const TriangleType & t) -> TransposedTriangleType { return {{t.a.x, t.b.x, t.c.x}, {t.a.y, t.b.y, t.c.y}, {t.a.z, t.b.z, t.c.z}}; };
    thrust::transform(triangleBegin, triangleEnd, transposedTriangleBegin, transposeTriangle);
    timer("setTriangle");  // 5.453ms
}
}  // namespace sah_kd_tree::helpers

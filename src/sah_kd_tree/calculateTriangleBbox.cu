#include "sah_kd_tree.cuh"

#include <thrust/advance.h>
#include <thrust/extrema.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <cassert>

namespace SahKdTree
{
template<I dimension>
void Projection<dimension>::calculateTriangleBbox()
{
    auto triangleCount = U(triangle.a.size());
    assert(triangleCount == U(triangle.b.size()));
    assert(triangleCount == U(triangle.c.size()));

    polygon.min.resize(triangleCount);
    polygon.max.resize(triangleCount);

    auto triangleBegin = thrust::make_zip_iterator(thrust::make_tuple(triangle.a.cbegin(), triangle.b.cbegin(), triangle.c.cbegin()));
    using TriangleType = IteratorValueType<decltype(triangleBegin)>;
    auto polygonBboxBegin = thrust::make_zip_iterator(thrust::make_tuple(polygon.min.begin(), polygon.max.begin()));
    using PolygonBboxType = IteratorValueType<decltype(polygonBboxBegin)>;
    auto toTriangleBbox = [] __host__ __device__(TriangleType triangle) -> PolygonBboxType {
        F a = thrust::get<0>(triangle);
        F b = thrust::get<1>(triangle);
        F c = thrust::get<2>(triangle);
        return {thrust::min(a, thrust::min(b, c)), thrust::max(a, thrust::max(b, c))};
    };
    thrust::transform(triangleBegin, thrust::next(triangleBegin, triangleCount), polygonBboxBegin, toTriangleBbox);
}

template void Projection<0>::calculateTriangleBbox();
template void Projection<1>::calculateTriangleBbox();
template void Projection<2>::calculateTriangleBbox();
}  // namespace SahKdTree

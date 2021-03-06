#include "sah_kd_tree/sah_kd_tree.hpp"
#include "sah_kd_tree/utility.cuh"

#include <thrust/advance.h>
#include <thrust/extrema.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/zip_function.h>

#include <cassert>

void sah_kd_tree::Projection::calculateTriangleBbox()
{
    auto triangleCount = U(triangle.a.size());
    assert(triangleCount == U(triangle.b.size()));
    assert(triangleCount == U(triangle.c.size()));

    polygon.min.resize(triangleCount);
    polygon.max.resize(triangleCount);

    auto triangleBegin = thrust::make_zip_iterator(thrust::make_tuple(triangle.a.cbegin(), triangle.b.cbegin(), triangle.c.cbegin()));
    auto polygonBboxBegin = thrust::make_zip_iterator(thrust::make_tuple(polygon.min.begin(), polygon.max.begin()));
    using PolygonBboxType = IteratorValueType<decltype(polygonBboxBegin)>;
    auto toTriangleBbox = [] __host__ __device__(F a, F b, F c) -> PolygonBboxType { return {thrust::min(a, thrust::min(b, c)), thrust::max(a, thrust::max(b, c))}; };
    thrust::transform(triangleBegin, thrust::next(triangleBegin, triangleCount), polygonBboxBegin, thrust::zip_function(toTriangleBbox));
}

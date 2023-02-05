#include <sah_kd_tree/sah_kd_tree.cuh>

#include <thrust/advance.h>
#include <thrust/extrema.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/zip_function.h>

void sah_kd_tree::Projection::calculateTriangleBbox()
{
    polygon.min.resize(triangle.count);
    polygon.max.resize(triangle.count);

    auto triangleBegin = thrust::make_zip_iterator(triangle.a, triangle.b, triangle.c);
    auto polygonBboxBegin = thrust::make_zip_iterator(polygon.min.begin(), polygon.max.begin());
    using PolygonBboxType = thrust::iterator_value_t<decltype(polygonBboxBegin)>;
    const auto toTriangleBbox = [] __host__ __device__(F a, F b, F c) -> PolygonBboxType
    {
        return {thrust::min(a, thrust::min(b, c)), thrust::max(a, thrust::max(b, c))};
    };
    thrust::transform(triangleBegin, thrust::next(triangleBegin, triangle.count), polygonBboxBegin, thrust::make_zip_function(toTriangleBbox));
}

#include <sah_kd_tree/sah_kd_tree.cuh>

#include <thrust/advance.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>
#include <thrust/zip_function.h>

auto sah_kd_tree::Builder::getSplittedPolygonCount(U layerBase, U layerSize) -> U
{
    auto nodePolygonCountBegin = thrust::make_zip_iterator(node.splitDimension.cbegin(), node.polygonCountLeft.cbegin(), node.polygonCountRight.cbegin(), node.polygonCount.cbegin());
    const auto toSplittedPolygonCount = [] __host__ __device__(I splitDimension, U polygonCountLeft, U polygonCountRight, U polygonCount) -> U {
        if (splitDimension < 0) {
            return 0;
        }
        return polygonCountLeft + polygonCountRight - polygonCount;
    };
    auto splittedPolygonCountBegin = thrust::next(nodePolygonCountBegin, layerBase);
    return thrust::transform_reduce(splittedPolygonCountBegin, thrust::next(splittedPolygonCountBegin, layerSize), thrust::make_zip_function(toSplittedPolygonCount), U(0), thrust::plus<U>{});
}

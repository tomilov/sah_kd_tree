#include "utility.cuh"

#include <sah_kd_tree/sah_kd_tree.hpp>

#include <thrust/advance.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>
#include <thrust/zip_function.h>

auto SahKdTree::Builder::getSplittedPolygonCount(U baseNode, U nodeCount) -> U
{
    auto nodePolygonCountBegin = thrust::make_zip_iterator(thrust::make_tuple(node.splitDimension.cbegin(), node.polygonCountLeft.cbegin(), node.polygonCountRight.cbegin(), node.polygonCount.cbegin()));
    auto toSplittedPolygonCount = [] __host__ __device__(I splitDimension, U polygonCountLeft, U polygonCountRight, U polygonCount) -> U {
        if (splitDimension < 0) {
            return 0;
        }
        return polygonCountLeft + polygonCountRight - polygonCount;
    };
    auto splittedPolygonCountBegin = thrust::next(nodePolygonCountBegin, baseNode);
    return thrust::transform_reduce(splittedPolygonCountBegin, thrust::next(splittedPolygonCountBegin, nodeCount), thrust::make_zip_function(toSplittedPolygonCount), U(0), thrust::plus<U>{});
}

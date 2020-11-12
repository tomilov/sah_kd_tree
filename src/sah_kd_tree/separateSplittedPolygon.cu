#include "utility.cuh"

#include <sah_kd_tree/sah_kd_tree.hpp>

#include <thrust/advance.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/zip_function.h>

#include <cassert>

void SahKdTree::Builder::separateSplittedPolygon(U baseNode, U polygonCount, U splittedPolygonCount)
{
    polygon.triangle.resize(polygonCount + splittedPolygonCount);
    polygon.node.resize(polygonCount + splittedPolygonCount);
    splittedPolygon.resize(splittedPolygonCount);

    auto polygonBegin = thrust::make_zip_iterator(thrust::make_tuple(polygon.triangle.begin(), polygon.node.begin()));
    auto indexedPolygonBegin = thrust::make_zip_iterator(thrust::make_tuple(thrust::make_counting_iterator<U>(0), polygonBegin));
    auto splitDimensionBegin = thrust::make_permutation_iterator(node.splitDimension.cbegin(), polygon.node.cbegin());
    auto splittedPolygonStencilBegin = thrust::make_zip_iterator(thrust::make_tuple(polygon.node.cbegin(), splitDimensionBegin, polygon.side.cbegin()));
    auto splittedPolygonBegin = thrust::make_zip_iterator(thrust::make_tuple(splittedPolygon.begin(), thrust::next(polygonBegin, polygonCount)));
    auto isSplittedPolygon = [baseNode] __host__ __device__(U polygonNode, I splitDimension, I polygonSide) -> bool {
        if (polygonNode < baseNode) {
            return false;
        }
        if (splitDimension < 0) {
            return false;
        }
        if (polygonSide != 0) {
            return false;
        }
        return true;
    };
    [[maybe_unused]] auto splittedPolygonEnd = thrust::copy_if(indexedPolygonBegin, thrust::next(indexedPolygonBegin, polygonCount), splittedPolygonStencilBegin, splittedPolygonBegin, thrust::zip_function(isSplittedPolygon));
    assert(thrust::next(splittedPolygonBegin, splittedPolygonCount) == splittedPolygonEnd);
}

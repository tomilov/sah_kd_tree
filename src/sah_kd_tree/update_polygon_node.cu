#include <sah_kd_tree/sah_kd_tree.cuh>
#include <sah_kd_tree/utility.cuh>

#include <thrust/advance.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/zip_function.h>

void sah_kd_tree::Builder::updatePolygonNode(U layerBase)
{
    auto nodeBothBegin = thrust::make_zip_iterator(node.nodeLeft.cbegin(), node.nodeRight.cbegin());
    auto polygonNodeBothBegin = thrust::make_permutation_iterator(nodeBothBegin, polygon.node.cbegin());
    using PolygonNodeBothType = IteratorValueType<decltype(polygonNodeBothBegin)>;
    auto toPolygonNode = [] __host__ __device__(I polygonSide, PolygonNodeBothType polygonNodeBoth) -> U {
        return (0 < polygonSide) ? thrust::get<1>(polygonNodeBoth) : thrust::get<0>(polygonNodeBoth);  // splitted polygon assigned to left node
    };
    auto splitDimensionBegin = thrust::make_permutation_iterator(node.splitDimension.cbegin(), polygon.node.cbegin());
    auto nodeStencilBegin = thrust::make_zip_iterator(polygon.node.cbegin(), splitDimensionBegin);
    auto isCurrentLayer = [layerBase] __host__ __device__(U polygonNode, I splitDimension) -> bool {
        if (polygonNode < layerBase) {
            return false;
        }
        if (splitDimension < 0) {
            return false;
        }
        // assert(!(polygonNode < layerBase)); ???
        return true;
    };
    thrust::transform_if(polygon.side.cbegin(), polygon.side.cend(), polygonNodeBothBegin, nodeStencilBegin, polygon.node.begin(), toPolygonNode, thrust::zip_function(isCurrentLayer));
}

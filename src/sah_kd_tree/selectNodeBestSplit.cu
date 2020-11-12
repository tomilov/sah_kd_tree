#include "utility.cuh"

#include <sah_kd_tree/sah_kd_tree.hpp>

#include <thrust/advance.h>
#include <thrust/extrema.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/zip_function.h>

#include <cassert>

void SahKdTree::Builder::selectNodeBestSplit(const Params & sah, U baseNode, U nodeCount)
{
    auto nodeSplitCostBegin = thrust::make_zip_iterator(thrust::make_tuple(x.layer.splitCost.cbegin(), y.layer.splitCost.cbegin(), z.layer.splitCost.cbegin()));
    using NodeBestCostType = IteratorValueType<decltype(nodeSplitCostBegin)>;
    auto nodeSplitPosBegin = thrust::make_zip_iterator(thrust::make_tuple(x.layer.splitPos.cbegin(), y.layer.splitPos.cbegin(), z.layer.splitPos.cbegin()));
    using NodeBestPosType = IteratorValueType<decltype(nodeSplitPosBegin)>;
    auto nodeLeftPolygonCountBegin = thrust::make_zip_iterator(thrust::make_tuple(x.layer.polygonCountLeft.cbegin(), y.layer.polygonCountLeft.cbegin(), z.layer.polygonCountLeft.cbegin()));
    using NodeLeftPolygonCountType = IteratorValueType<decltype(nodeLeftPolygonCountBegin)>;
    auto nodeRightPolygonCountBegin = thrust::make_zip_iterator(thrust::make_tuple(x.layer.polygonCountRight.cbegin(), y.layer.polygonCountRight.cbegin(), z.layer.polygonCountRight.cbegin()));
    using NodeRightPolygonCountType = IteratorValueType<decltype(nodeRightPolygonCountBegin)>;
    auto nodePolygonCountBegin = thrust::next(node.polygonCount.cbegin(), baseNode);
    auto nodeSplitBegin = thrust::make_zip_iterator(thrust::make_tuple(nodeSplitCostBegin, nodeSplitPosBegin, nodeLeftPolygonCountBegin, nodeRightPolygonCountBegin, nodePolygonCountBegin));
    auto nodeBestSplitBegin = thrust::next(thrust::make_zip_iterator(thrust::make_tuple(node.splitDimension.begin(), node.splitPos.begin(), node.polygonCountLeft.begin(), node.polygonCountRight.begin())), baseNode);
    using NodeBestSplitType = IteratorValueType<decltype(nodeBestSplitBegin)>;
    auto toNodeBestSplit = [sah] __host__ __device__(NodeBestCostType nodeSplitCost, NodeBestPosType nodeSplitPos, NodeLeftPolygonCountType nodeLeftPolygonCount, NodeRightPolygonCountType nodeRightPolygonCount,
                                                     U nodePolygonCount) -> NodeBestSplitType {
        assert(nodePolygonCount != 0);
        F x = thrust::get<0>(nodeSplitCost);
        F y = thrust::get<1>(nodeSplitCost);
        F z = thrust::get<2>(nodeSplitCost);
        F bestNodeSplitCost = thrust::min(sah.intersectionCost * nodePolygonCount, thrust::min(x, thrust::min(y, z)));
        if (!(bestNodeSplitCost < x)) {
            return {0, thrust::get<0>(nodeSplitPos), thrust::get<0>(nodeLeftPolygonCount), thrust::get<0>(nodeRightPolygonCount)};
        } else if (!(bestNodeSplitCost < y)) {
            return {1, thrust::get<1>(nodeSplitPos), thrust::get<1>(nodeLeftPolygonCount), thrust::get<1>(nodeRightPolygonCount)};
        } else if (!(bestNodeSplitCost < z)) {
            return {2, thrust::get<2>(nodeSplitPos), thrust::get<2>(nodeLeftPolygonCount), thrust::get<2>(nodeRightPolygonCount)};
        } else {
            return {-1};  // terminate
        }
    };
    auto isNodeNotEmpty = [] __host__ __device__(U nodePolygonCount) -> bool { return nodePolygonCount != 0; };
    thrust::transform_if(nodeSplitBegin, thrust::next(nodeSplitBegin, nodeCount), nodePolygonCountBegin, nodeBestSplitBegin, thrust::make_zip_function(toNodeBestSplit), isNodeNotEmpty);
}

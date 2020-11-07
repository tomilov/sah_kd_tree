#include "utility.cuh"

#include <sah_kd_tree/builder.hpp>

#include <thrust/advance.h>
#include <thrust/extrema.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <cassert>

void SahKdTree::Builder::selectNodeBestSplit(const Params & sah, U baseNode, U nodeCount)
{
    Timer timer;
    auto nodeSplitCostBegin = thrust::make_zip_iterator(thrust::make_tuple(x.layer.splitCost.cbegin(), y.layer.splitCost.cbegin(), z.layer.splitCost.cbegin()));
    auto nodeSplitPosBegin = thrust::make_zip_iterator(thrust::make_tuple(x.layer.splitPos.cbegin(), y.layer.splitPos.cbegin(), z.layer.splitPos.cbegin()));
    auto nodeLeftPolygonCountBegin = thrust::make_zip_iterator(thrust::make_tuple(x.layer.polygonCountLeft.cbegin(), y.layer.polygonCountLeft.cbegin(), z.layer.polygonCountLeft.cbegin()));
    auto nodeRightPolygonCountBegin = thrust::make_zip_iterator(thrust::make_tuple(x.layer.polygonCountRight.cbegin(), y.layer.polygonCountRight.cbegin(), z.layer.polygonCountRight.cbegin()));
    auto nodeSplitBegin = thrust::make_zip_iterator(thrust::make_tuple(nodeSplitCostBegin, nodeSplitPosBegin, nodeLeftPolygonCountBegin, nodeRightPolygonCountBegin));
    using NodeSplitType = IteratorValueType<decltype(nodeSplitBegin)>;
    auto nodePolygonCountBegin = thrust::next(node.polygonCount.cbegin(), baseNode);
    auto nodeBestSplitBegin = thrust::next(thrust::make_zip_iterator(thrust::make_tuple(node.splitDimension.begin(), node.splitPos.begin(), node.polygonCountLeft.begin(), node.polygonCountRight.begin())), baseNode);
    using NodeBestSplitType = IteratorValueType<decltype(nodeBestSplitBegin)>;
    auto toNodeBestSplit = [sah] __host__ __device__(NodeSplitType nodeSplit, U nodePolygonCount) -> NodeBestSplitType {
        assert(nodePolygonCount != 0);
        const auto & nodeSplitCost = thrust::get<0>(nodeSplit);
        const auto & nodeSplitPos = thrust::get<1>(nodeSplit);
        const auto & nodeLeftPolygonCount = thrust::get<2>(nodeSplit);
        const auto & nodeRightPolygonCount = thrust::get<3>(nodeSplit);
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
    thrust::transform_if(nodeSplitBegin, thrust::next(nodeSplitBegin, nodeCount), nodePolygonCountBegin, nodePolygonCountBegin, nodeBestSplitBegin, toNodeBestSplit, isNodeNotEmpty);
    timer("selectNodeBestSplit");  // 0.000017
}

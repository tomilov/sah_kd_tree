#include <sah_kd_tree/sah_kd_tree.cuh>
#include <sah_kd_tree/utility.cuh>

#include <thrust/advance.h>
#include <thrust/extrema.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/zip_function.h>

#include <cassert>

void sah_kd_tree::Builder::selectNodeBestSplit(const Params & sah, U layerBase, U layerSize)
{
    auto nodeSplitCostBegin = thrust::make_zip_iterator(x.layer.splitCost.cbegin(), y.layer.splitCost.cbegin(), z.layer.splitCost.cbegin());
    using NodeBestCostType = IteratorValueType<decltype(nodeSplitCostBegin)>;
    auto nodeSplitPosBegin = thrust::make_zip_iterator(x.layer.splitPos.cbegin(), y.layer.splitPos.cbegin(), z.layer.splitPos.cbegin());
    using NodeBestPosType = IteratorValueType<decltype(nodeSplitPosBegin)>;
    auto nodeLeftPolygonCountBegin = thrust::make_zip_iterator(x.layer.polygonCountLeft.cbegin(), y.layer.polygonCountLeft.cbegin(), z.layer.polygonCountLeft.cbegin());
    using NodePolygonCountType = IteratorValueType<decltype(nodeLeftPolygonCountBegin)>;
    auto nodeRightPolygonCountBegin = thrust::make_zip_iterator(x.layer.polygonCountRight.cbegin(), y.layer.polygonCountRight.cbegin(), z.layer.polygonCountRight.cbegin());
    using NodePolygonCountType = IteratorValueType<decltype(nodeRightPolygonCountBegin)>;
    auto nodePolygonCountBegin = thrust::next(node.polygonCount.cbegin(), layerBase);
    auto nodeSplitBegin = thrust::make_zip_iterator(nodeSplitCostBegin, nodeSplitPosBegin, nodeLeftPolygonCountBegin, nodeRightPolygonCountBegin, nodePolygonCountBegin);
    auto nodeBestSplitBegin = thrust::make_zip_iterator(node.splitDimension.begin(), node.splitPos.begin(), node.polygonCountLeft.begin(), node.polygonCountRight.begin());
    thrust::advance(nodeBestSplitBegin, layerBase);
    using NodeBestSplitType = IteratorValueType<decltype(nodeBestSplitBegin)>;
    const auto toNodeBestSplit = [sah] __host__ __device__(NodeBestCostType nodeSplitCost, NodeBestPosType nodeSplitPos, NodePolygonCountType nodeLeftPolygonCount, NodePolygonCountType nodeRightPolygonCount, U nodePolygonCount) -> NodeBestSplitType {
        assert(nodePolygonCount != 0);
        F x = thrust::get<0>(nodeSplitCost);
        F y = thrust::get<1>(nodeSplitCost);
        F z = thrust::get<2>(nodeSplitCost);
        F t = sah.intersectionCost * nodePolygonCount;
        F bestNodeSplitCost = thrust::min(t, thrust::min(x, thrust::min(y, z)));
        if (!(bestNodeSplitCost < x)) {
            return {0, thrust::get<0>(nodeSplitPos), thrust::get<0>(nodeLeftPolygonCount), thrust::get<0>(nodeRightPolygonCount)};
        } else if (!(bestNodeSplitCost < y)) {
            return {1, thrust::get<1>(nodeSplitPos), thrust::get<1>(nodeLeftPolygonCount), thrust::get<1>(nodeRightPolygonCount)};
        } else if (!(bestNodeSplitCost < z)) {
            return {2, thrust::get<2>(nodeSplitPos), thrust::get<2>(nodeLeftPolygonCount), thrust::get<2>(nodeRightPolygonCount)};
        } else {
            assert(!(bestNodeSplitCost < t));
            return {-1};  // terminate
        }
    };
    const auto isNodeNotEmpty = [] __host__ __device__(U nodePolygonCount) -> bool { return nodePolygonCount != 0; };
    thrust::transform_if(nodeSplitBegin, thrust::next(nodeSplitBegin, layerSize), nodePolygonCountBegin, nodeBestSplitBegin, thrust::make_zip_function(toNodeBestSplit), isNodeNotEmpty);
}

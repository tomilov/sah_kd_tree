#include <sah_kd_tree/sah_kd_tree.cuh>
#include <sah_kd_tree/type_traits.cuh>

#include <thrust/advance.h>
#include <thrust/extrema.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/zip_function.h>

#include <cassert>

void sah_kd_tree::Builder::selectNodeBestSplit(const Params & sah, const Projection & x, const Projection & y, const Projection & z)
{
    auto nodeSplitCostBegin = thrust::make_zip_iterator(x.layer.splitCost.cbegin(), y.layer.splitCost.cbegin(), z.layer.splitCost.cbegin());
    using NodeSplitCostType = IteratorValueType<decltype(nodeSplitCostBegin)>;
    auto nodeSplitPosBegin = thrust::make_zip_iterator(x.layer.splitPos.cbegin(), y.layer.splitPos.cbegin(), z.layer.splitPos.cbegin());
    using NodeSplitPosType = IteratorValueType<decltype(nodeSplitPosBegin)>;
    auto nodeLeftChildPolygonCountBegin = thrust::make_zip_iterator(x.layer.polygonCountLeft.cbegin(), y.layer.polygonCountLeft.cbegin(), z.layer.polygonCountLeft.cbegin());
    using NodePolygonCountType = IteratorValueType<decltype(nodeLeftChildPolygonCountBegin)>;
    auto nodeRightChildPolygonCountBegin = thrust::make_zip_iterator(x.layer.polygonCountRight.cbegin(), y.layer.polygonCountRight.cbegin(), z.layer.polygonCountRight.cbegin());
    using NodePolygonCountType = IteratorValueType<decltype(nodeRightChildPolygonCountBegin)>;
    auto nodePolygonCountBegin = thrust::next(node.polygonCount.cbegin(), layer.base);
    auto nodeSplitBegin = thrust::make_zip_iterator(nodeSplitCostBegin, nodeSplitPosBegin, nodeLeftChildPolygonCountBegin, nodeRightChildPolygonCountBegin, nodePolygonCountBegin);
    auto nodeBestSplitBegin = thrust::make_zip_iterator(node.splitDimension.begin(), node.splitPos.begin(), node.polygonCountLeft.begin(), node.polygonCountRight.begin());
    thrust::advance(nodeBestSplitBegin, layer.base);
    using NodeBestSplitType = IteratorValueType<decltype(nodeBestSplitBegin)>;
    const auto toNodeBestSplit = [sah] __host__ __device__(NodeSplitCostType nodeSplitCost, NodeSplitPosType nodeSplitPos, NodePolygonCountType nodeLeftChildPolygonCount, NodePolygonCountType nodeRightChildPolygonCount,
                                                           U nodePolygonCount) -> NodeBestSplitType {
        assert(nodePolygonCount != 0);
        thrust::tuple<F, U> x{thrust::get<0>(nodeSplitCost), thrust::get<0>(nodeLeftChildPolygonCount) + thrust::get<0>(nodeRightChildPolygonCount) - nodePolygonCount};
        thrust::tuple<F, U> y{thrust::get<1>(nodeSplitCost), thrust::get<1>(nodeLeftChildPolygonCount) + thrust::get<1>(nodeRightChildPolygonCount) - nodePolygonCount};
        thrust::tuple<F, U> z{thrust::get<2>(nodeSplitCost), thrust::get<2>(nodeLeftChildPolygonCount) + thrust::get<2>(nodeRightChildPolygonCount) - nodePolygonCount};
        thrust::tuple<F, U> t{sah.intersectionCost * nodePolygonCount, 0};
        thrust::tuple<F, U> bestNodeSplitCost = thrust::min(t, thrust::min(x, thrust::min(y, z)));
        if (!(bestNodeSplitCost < x)) {
            return {0, thrust::get<0>(nodeSplitPos), thrust::get<0>(nodeLeftChildPolygonCount), thrust::get<0>(nodeRightChildPolygonCount)};
        } else if (!(bestNodeSplitCost < y)) {
            return {1, thrust::get<1>(nodeSplitPos), thrust::get<1>(nodeLeftChildPolygonCount), thrust::get<1>(nodeRightChildPolygonCount)};
        } else if (!(bestNodeSplitCost < z)) {
            return {2, thrust::get<2>(nodeSplitPos), thrust::get<2>(nodeLeftChildPolygonCount), thrust::get<2>(nodeRightChildPolygonCount)};
        } else {
            assert(!(bestNodeSplitCost < t));
            return {-1};  // leaf
        }
    };
    thrust::transform_if(nodeSplitBegin, thrust::next(nodeSplitBegin, layer.size), nodePolygonCountBegin, nodeBestSplitBegin, thrust::make_zip_function(toNodeBestSplit), isNodeNotEmpty);
}

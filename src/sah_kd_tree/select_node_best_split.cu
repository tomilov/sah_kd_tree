#include <sah_kd_tree/sah_kd_tree.cuh>
#include <sah_kd_tree/type_traits.cuh>

#include <thrust/advance.h>
#include <thrust/extrema.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <cassert>

void sah_kd_tree::Builder::selectNodeBestSplit(const Params & sah, const Projection & x, const Projection & y, const Projection & z)
{
    auto layerNodeBegin = thrust::make_counting_iterator<U>(0);
    auto layerNodeEnd = thrust::make_counting_iterator<U>(layer.size);

    auto nodeXSplitCosts = x.layer.splitCost.data().get();
    auto nodeYSplitCosts = y.layer.splitCost.data().get();
    auto nodeZSplitCosts = z.layer.splitCost.data().get();

    auto nodeXLeftChildPolygonCounts = x.layer.polygonCountLeft.data().get();
    auto nodeYLeftChildPolygonCounts = y.layer.polygonCountLeft.data().get();
    auto nodeZLeftChildPolygonCounts = z.layer.polygonCountLeft.data().get();

    auto nodeXRightChildPolygonCounts = x.layer.polygonCountRight.data().get();
    auto nodeYRightChildPolygonCounts = y.layer.polygonCountRight.data().get();
    auto nodeZRightChildPolygonCounts = z.layer.polygonCountRight.data().get();

    auto nodePolygonCounts = thrust::next(node.polygonCount.data(), layer.base).get();

    auto nodeXSplitPositions = x.layer.splitPos.data().get();
    auto nodeYSplitPositions = y.layer.splitPos.data().get();
    auto nodeZSplitPositions = z.layer.splitPos.data().get();

    auto nodeBestSplitBegin = thrust::make_zip_iterator(node.splitDimension.begin(), node.splitPos.begin(), node.polygonCountLeft.begin(), node.polygonCountRight.begin());
    using NodeBestSplitType = IteratorValueType<decltype(nodeBestSplitBegin)>;
    const auto toNodeBestSplit = [sah, nodeXSplitCosts, nodeYSplitCosts, nodeZSplitCosts, nodeXLeftChildPolygonCounts, nodeYLeftChildPolygonCounts, nodeZLeftChildPolygonCounts, nodeXRightChildPolygonCounts, nodeYRightChildPolygonCounts,
                                  nodeZRightChildPolygonCounts, nodePolygonCounts, nodeXSplitPositions, nodeYSplitPositions, nodeZSplitPositions] __host__
                                 __device__(U layerNode) -> NodeBestSplitType {
        U nodePolygonCount = nodePolygonCounts[layerNode];
        assert(nodePolygonCount != 0);

        auto nodeXLeftChildPolygonCount = nodeXLeftChildPolygonCounts[layerNode];
        auto nodeYLeftChildPolygonCount = nodeYLeftChildPolygonCounts[layerNode];
        auto nodeZLeftChildPolygonCount = nodeZLeftChildPolygonCounts[layerNode];

        auto nodeXRightChildPolygonCount = nodeXRightChildPolygonCounts[layerNode];
        auto nodeYRightChildPolygonCount = nodeYRightChildPolygonCounts[layerNode];
        auto nodeZRightChildPolygonCount = nodeZRightChildPolygonCounts[layerNode];

        thrust::tuple<F, U> x{nodeXSplitCosts[layerNode], nodeXLeftChildPolygonCount + nodeXRightChildPolygonCount - nodePolygonCount};
        thrust::tuple<F, U> y{nodeYSplitCosts[layerNode], nodeYLeftChildPolygonCount + nodeYRightChildPolygonCount - nodePolygonCount};
        thrust::tuple<F, U> z{nodeZSplitCosts[layerNode], nodeZLeftChildPolygonCount + nodeZRightChildPolygonCount - nodePolygonCount};

        thrust::tuple<F, U> t{sah.intersectionCost * nodePolygonCount, 0};

        thrust::tuple<F, U> bestNodeSplitCost = thrust::min(t, thrust::min(x, thrust::min(y, z)));
        if (!(bestNodeSplitCost < x)) {
            return {0, nodeXSplitPositions[layerNode], nodeXLeftChildPolygonCount, nodeXRightChildPolygonCount};
        } else if (!(bestNodeSplitCost < y)) {
            return {1, nodeYSplitPositions[layerNode], nodeYLeftChildPolygonCount, nodeYRightChildPolygonCount};
        } else if (!(bestNodeSplitCost < z)) {
            return {2, nodeZSplitPositions[layerNode], nodeZLeftChildPolygonCount, nodeZRightChildPolygonCount};
        } else {
            assert(!(bestNodeSplitCost < t));
            return {-1};  // leaf
        }
    };
    thrust::transform_if(layerNodeBegin, layerNodeEnd, nodePolygonCounts, thrust::next(nodeBestSplitBegin, layer.base), toNodeBestSplit, isNodeNotEmpty);
}

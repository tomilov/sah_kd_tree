#include <sah_kd_tree/sah_kd_tree.cuh>

#include <thrust/advance.h>
#include <thrust/extrema.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/tabulate_output_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/memory.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <cassert>

template<typename Traits>
void sah_kd_tree::Builder<Traits>::selectNodeBestSplit(const Params<Traits> & sah, const Projection<Traits> & x, const Projection<Traits> & y, const Projection<Traits> & z)
{
    auto layerNodeBegin = thrust::make_counting_iterator<U>(0);
    auto layerNodeEnd = thrust::make_counting_iterator<U>(layer.size);

    auto nodePolygonCountBegin = cuda::std::next(node.polygonCount.cbegin(), layer.base);

    auto nodeXSplitCosts = thrust::raw_pointer_cast(x.layer.splitCost.data());
    auto nodeYSplitCosts = thrust::raw_pointer_cast(y.layer.splitCost.data());
    auto nodeZSplitCosts = thrust::raw_pointer_cast(z.layer.splitCost.data());

    auto nodeXLeftChildPolygonCounts = thrust::raw_pointer_cast(x.layer.polygonCountLeft.data());
    auto nodeYLeftChildPolygonCounts = thrust::raw_pointer_cast(y.layer.polygonCountLeft.data());
    auto nodeZLeftChildPolygonCounts = thrust::raw_pointer_cast(z.layer.polygonCountLeft.data());

    auto nodeXRightChildPolygonCounts = thrust::raw_pointer_cast(x.layer.polygonCountRight.data());
    auto nodeYRightChildPolygonCounts = thrust::raw_pointer_cast(y.layer.polygonCountRight.data());
    auto nodeZRightChildPolygonCounts = thrust::raw_pointer_cast(z.layer.polygonCountRight.data());

    auto nodePolygonCounts = thrust::raw_pointer_cast(cuda::std::next(node.polygonCount.data(), layer.base));

    auto nodeXSplitPositions = thrust::raw_pointer_cast(x.layer.splitPos.data());
    auto nodeYSplitPositions = thrust::raw_pointer_cast(y.layer.splitPos.data());
    auto nodeZSplitPositions = thrust::raw_pointer_cast(z.layer.splitPos.data());

    auto nodeBestSplitBegin = thrust::make_zip_iterator(node.splitDimension.begin(), node.splitPos.begin(), node.polygonCountLeft.begin(), node.polygonCountRight.begin());
    using NodeBestSplitType = cuda::std::iter_value_t<decltype(nodeBestSplitBegin)>;
    const auto toNodeBestSplit = [sah, nodeXSplitCosts, nodeYSplitCosts, nodeZSplitCosts, nodeXLeftChildPolygonCounts, nodeYLeftChildPolygonCounts, nodeZLeftChildPolygonCounts, nodeXRightChildPolygonCounts, nodeYRightChildPolygonCounts,
                                  nodeZRightChildPolygonCounts, nodePolygonCounts, nodeXSplitPositions, nodeYSplitPositions, nodeZSplitPositions] __host__
                                 __device__(U layerNode) -> NodeBestSplitType
    {
        U nodePolygonCount = nodePolygonCounts[layerNode];

        auto nodeXLeftChildPolygonCount = nodeXLeftChildPolygonCounts[layerNode];
        auto nodeYLeftChildPolygonCount = nodeYLeftChildPolygonCounts[layerNode];
        auto nodeZLeftChildPolygonCount = nodeZLeftChildPolygonCounts[layerNode];

        auto nodeXRightChildPolygonCount = nodeXRightChildPolygonCounts[layerNode];
        auto nodeYRightChildPolygonCount = nodeYRightChildPolygonCounts[layerNode];
        auto nodeZRightChildPolygonCount = nodeZRightChildPolygonCounts[layerNode];

        cuda::std::tuple<F, U> x{nodeXSplitCosts[layerNode], nodeXLeftChildPolygonCount + nodeXRightChildPolygonCount - nodePolygonCount};
        cuda::std::tuple<F, U> y{nodeYSplitCosts[layerNode], nodeYLeftChildPolygonCount + nodeYRightChildPolygonCount - nodePolygonCount};
        cuda::std::tuple<F, U> z{nodeZSplitCosts[layerNode], nodeZLeftChildPolygonCount + nodeZRightChildPolygonCount - nodePolygonCount};

        cuda::std::tuple<F, U> t{sah.intersectionCost * static_cast<F>(nodePolygonCount), 0};

        cuda::std::tuple<F, U> bestNodeSplitCost = thrust::min(t, thrust::min(x, thrust::min(y, z)));
        if (!(bestNodeSplitCost < x)) {
            return {0, nodeXSplitPositions[layerNode], nodeXLeftChildPolygonCount, nodeXRightChildPolygonCount};
        } else if (!(bestNodeSplitCost < y)) {
            return {1, nodeYSplitPositions[layerNode], nodeYLeftChildPolygonCount, nodeYRightChildPolygonCount};
        } else if (!(bestNodeSplitCost < z)) {
            return {2, nodeZSplitPositions[layerNode], nodeZLeftChildPolygonCount, nodeZRightChildPolygonCount};
        } else {
            assert(!(bestNodeSplitCost < t));
            return NodeBestSplitType{-1};  // leaf node
        }
    };
    thrust::transform_if(layerNodeBegin, layerNodeEnd, nodePolygonCountBegin, cuda::std::next(nodeBestSplitBegin, layer.base), toNodeBestSplit, isNodeNotEmpty);
}

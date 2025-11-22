#include <sah_kd_tree/sah_kd_tree.cuh>

#include <thrust/advance.h>
#include <thrust/count.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/transform_scan.h>

#include <utility>

#include <cassert>

template<typename Traits>
template<typename P>
bool sah_kd_tree::Builder<Traits>::build(const P & progress, const Params<Traits> & sah, Projection<Traits> & x, Projection<Traits> & y, Projection<Traits> & z, Tree<Traits> & tree)
{
    x.calculateTriangleBbox();
    y.calculateTriangleBbox();
    z.calculateTriangleBbox();

    x.calculateRootNodeBbox();
    y.calculateRootNodeBbox();
    z.calculateRootNodeBbox();

    x.generateInitialEvent();
    y.generateInitialEvent();
    z.generateInitialEvent();

    polygon.triangle.resize(polygon.count);
    thrust::sequence(polygon.triangle.begin(), polygon.triangle.end());  // TODO(tomilov): do not waste space
    polygon.node.resize(polygon.count, static_cast<U>(0));

    node.polygonCount.resize(1, polygon.count);
    resizeNode();

    layer.nodeOffset.resize(1, static_cast<U>(0));

    tree.layerDepth.push_back(node.count);
    for (;;) {
        assert(!std::empty(layer.nodeOffset));

        if (progress(tree.layerDepth.size())) {
            return false;
        }

        if (tree.layerDepth.size() == sah.maxTreeDepth) {
            leaf.count += layer.size;
            break;
        }

        x.findPerfectSplit(sah, layer.size, layer.nodeOffset, node.polygonCount, y, z);
        y.findPerfectSplit(sah, layer.size, layer.nodeOffset, node.polygonCount, z, x);
        z.findPerfectSplit(sah, layer.size, layer.nodeOffset, node.polygonCount, x, y);
        selectNodeBestSplit(sah, x, y, z);

        auto layerSplitDimensionBegin = thrust::next(node.splitDimension.cbegin(), layer.base);
        auto layerSplitDimensionEnd = thrust::next(layerSplitDimensionBegin, layer.size);
        assert(layerSplitDimensionEnd == node.splitDimension.cend());
        U layerLeafNodeCount = safeConvert<U>(thrust::count(layerSplitDimensionBegin, layerSplitDimensionEnd, kNoSplitDimension));
        leaf.count += layerLeafNodeCount;
        if (layerLeafNodeCount == layer.size) {
            assert(tree.layerDepth.size() < sah.maxTreeDepth);
            break;
        }
        assert(layerLeafNodeCount < layer.size);

        polygon.side.resize(polygon.count);
        polygon.eventRight.resize(polygon.count);

        determinePolygonSide<0>(x);
        determinePolygonSide<1>(y);
        determinePolygonSide<2>(z);

        updateSplittedPolygonCount();

        {  // generate index for child node pair
            auto nodeLeftChildBegin = thrust::next(node.leftChild.begin(), layer.base);
            const auto toNodeCount = [] __host__ __device__(I layerSplitDimension) -> U
            {
                return (layerSplitDimension < 0) ? 0 : 2;
            };
            auto nodeLeftChildEnd = thrust::transform_exclusive_scan(layerSplitDimensionBegin, layerSplitDimensionEnd, nodeLeftChildBegin, toNodeCount, layer.base + layer.size, cuda::std::plus<U>{});

            auto nodeRightChildBegin = thrust::next(node.rightChild.begin(), layer.base);
            const auto toNodeRightChild = [] __host__ __device__(U nodeLeftChild, I layerSplitDimension) -> U
            {
                return (layerSplitDimension < 0) ? 0 : (nodeLeftChild + 1);
            };
            thrust::transform(nodeLeftChildBegin, nodeLeftChildEnd, layerSplitDimensionBegin, nodeRightChildBegin, toNodeRightChild);
        }

        separateSplittedPolygon();

        x.decoupleEventBoth(node.splitDimension, polygon.side);
        y.decoupleEventBoth(node.splitDimension, polygon.side);
        z.decoupleEventBoth(node.splitDimension, polygon.side);

        assert(polygon.side.size() == polygon.count);
        updatePolygonNode();

        splitPolygon<0>(x, y, z);
        splitPolygon<1>(y, z, x);
        splitPolygon<2>(z, x, y);

        updateSplittedPolygonNode();

        x.mergeEvent(polygon.count, polygon.splittedCount, polygon.node, splittedPolygon);
        y.mergeEvent(polygon.count, polygon.splittedCount, polygon.node, splittedPolygon);
        z.mergeEvent(polygon.count, polygon.splittedCount, polygon.node, splittedPolygon);

        U layerBasePrev = layer.base;
        layer.base += layer.size;
        layer.size -= layerLeafNodeCount;
        layer.size += layer.size;

        node.count = layer.base + layer.size;

        node.polygonCount.resize(node.count);

        auto nodePolygonCountLeftBegin = thrust::next(node.polygonCountLeft.cbegin(), layerBasePrev);
        auto nodePolygonCountLeftEnd = thrust::next(node.polygonCountLeft.cbegin(), layer.base);
        thrust::scatter_if(nodePolygonCountLeftBegin, nodePolygonCountLeftEnd, thrust::next(node.leftChild.cbegin(), layerBasePrev), layerSplitDimensionBegin, node.polygonCount.begin(), isNotLeaf);
        auto nodePolygonCountRightBegin = thrust::next(node.polygonCountRight.cbegin(), layerBasePrev);
        auto nodePolygonCountRightEnd = thrust::next(node.polygonCountRight.cbegin(), layer.base);
        thrust::scatter_if(nodePolygonCountRightBegin, nodePolygonCountRightEnd, thrust::next(node.rightChild.cbegin(), layerBasePrev), layerSplitDimensionBegin, node.polygonCount.begin(), isNotLeaf);

        setNodeCount(x, y, z);

        auto nodeBboxBegin = thrust::make_zip_iterator(x.node.min.begin(), x.node.max.begin(), y.node.min.begin(), y.node.max.begin(), z.node.min.begin(), z.node.max.begin());
        auto layerBboxBegin = thrust::next(nodeBboxBegin, layerBasePrev);
        auto layerBboxEnd = thrust::next(nodeBboxBegin, layer.base);
        thrust::scatter_if(layerBboxBegin, layerBboxEnd, thrust::next(node.leftChild.cbegin(), layerBasePrev), layerSplitDimensionBegin, nodeBboxBegin, isNotLeaf);
        thrust::scatter_if(layerBboxBegin, layerBboxEnd, thrust::next(node.rightChild.cbegin(), layerBasePrev), layerSplitDimensionBegin, nodeBboxBegin, isNotLeaf);

        splitNode<0>(layerBasePrev, x);
        splitNode<1>(layerBasePrev, y);
        splitNode<2>(layerBasePrev, z);
        polygon.count += polygon.splittedCount;

        resizeNode();
        tree.layerDepth.push_back(node.count);
        filterLayerNodeOffset();

        assert(checkBoxes(x, y, z));
    }

    populateNodeParent();
    assert(checkBoxes(x, y, z));
    populateLeafNodeTriangleRange();
    assert(checkNodes(x, y, z));

    calculateRope<0, false>(x, y, z);
    calculateRope<0, true>(x, y, z);

    calculateRope<1, false>(y, z, x);
    calculateRope<1, true>(y, z, x);

    calculateRope<2, false>(z, x, y);
    calculateRope<2, true>(z, x, y);

    {
        tree.x.node.min = std::move(x.node.min);
        tree.x.node.max = std::move(x.node.max);
        tree.x.node.leftRope = std::move(x.node.leftRope);
        tree.x.node.rightRope = std::move(x.node.rightRope);

        tree.y.node.min = std::move(y.node.min);
        tree.y.node.max = std::move(y.node.max);
        tree.y.node.leftRope = std::move(y.node.leftRope);
        tree.y.node.rightRope = std::move(y.node.rightRope);

        tree.z.node.min = std::move(z.node.min);
        tree.z.node.max = std::move(z.node.max);
        tree.z.node.leftRope = std::move(z.node.leftRope);
        tree.z.node.rightRope = std::move(z.node.rightRope);

        tree.node.splitDimension = std::move(node.splitDimension);
        tree.node.splitPos = std::move(node.splitPos);
        tree.node.leftChild = std::move(node.leftChild);
        tree.node.rightChild = std::move(node.rightChild);

        tree.polygonTriangle = std::move(polygon.triangle);

        tree.node.parent = std::move(node.parent);
    }

    return true;
}

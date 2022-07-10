#include <sah_kd_tree/sah_kd_tree.cuh>

#include <thrust/advance.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/transform_scan.h>

#include <cassert>

auto sah_kd_tree::Builder::operator()(const Params & sah, Projection & x, Projection & y, Projection & z) -> Tree
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
    thrust::sequence(polygon.triangle.begin(), polygon.triangle.end());
    polygon.node.assign(polygon.count, U(0));

    node.splitDimension.resize(1);
    node.splitPos.resize(1);
    node.leftChild.resize(1);
    node.rightChild.resize(1);
    node.polygonCount.assign(1, polygon.count);
    node.polygonCountLeft.resize(1);
    node.polygonCountRight.resize(1);

    Tree tree;
    for (tree.depth = 0; tree.depth < sah.maxDepth; ++tree.depth) {
        filterLayerNodeOffset();

        x.findPerfectSplit(sah, layer.size, layer.nodeOffset, node.polygonCount, y, z);
        y.findPerfectSplit(sah, layer.size, layer.nodeOffset, node.polygonCount, z, x);
        z.findPerfectSplit(sah, layer.size, layer.nodeOffset, node.polygonCount, x, y);

        selectNodeBestSplit(sah, x, y, z);

        auto layerSplitDimensionBegin = thrust::next(node.splitDimension.cbegin(), layer.base);
        auto layerSplitDimensionEnd = thrust::next(layerSplitDimensionBegin, layer.size);
        auto layerLeafNodeCount = U(thrust::count(layerSplitDimensionBegin, layerSplitDimensionEnd, I(-1)));
        node.leafCount += layerLeafNodeCount;
        if (layerLeafNodeCount == layer.size) {
            break;
        }

        polygon.side.resize(polygon.count);
        polygon.eventRight.resize(polygon.count);

        determinePolygonSide<0>(x);
        determinePolygonSide<1>(y);
        determinePolygonSide<2>(z);

        updateSplittedPolygonCount();

        {  // generate index for child node
            auto nodeLeftChildBegin = thrust::next(node.leftChild.begin(), layer.base);
            const auto toNodeCount = [] __host__ __device__(I layerSplitDimension) -> U { return (layerSplitDimension < 0) ? 0 : 2; };
            auto nodeLeftChildEnd = thrust::transform_exclusive_scan(layerSplitDimensionBegin, layerSplitDimensionEnd, nodeLeftChildBegin, toNodeCount, layer.base + layer.size, thrust::plus<U>{});

            auto nodeRightChildBegin = thrust::next(node.rightChild.begin(), layer.base);
            const auto toNodeRightChild = [] __host__ __device__(U nodeLeftChild) -> U { return nodeLeftChild + 1; };
            thrust::transform(nodeLeftChildBegin, nodeLeftChildEnd, nodeRightChildBegin, toNodeRightChild);
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
        thrust::scatter_if(nodePolygonCountLeftBegin, nodePolygonCountLeftEnd, thrust::next(node.leftChild.cbegin(), layerBasePrev), layerSplitDimensionBegin, node.polygonCount.begin(), IsNotLeaf{});
        auto nodePolygonCountRightBegin = thrust::next(node.polygonCountRight.cbegin(), layerBasePrev);
        auto nodePolygonCountRightEnd = thrust::next(node.polygonCountRight.cbegin(), layer.base);
        thrust::scatter_if(nodePolygonCountRightBegin, nodePolygonCountRightEnd, thrust::next(node.rightChild.cbegin(), layerBasePrev), layerSplitDimensionBegin, node.polygonCount.begin(), IsNotLeaf{});

        setNodeCount(x, y, z);

        auto nodeBboxBegin = thrust::make_zip_iterator(x.node.min.begin(), x.node.max.begin(), y.node.min.begin(), y.node.max.begin(), z.node.min.begin(), z.node.max.begin());
        auto layerBboxBegin = thrust::next(nodeBboxBegin, layerBasePrev);
        auto layerBboxEnd = thrust::next(nodeBboxBegin, layer.base);
        thrust::scatter_if(layerBboxBegin, layerBboxEnd, thrust::next(node.leftChild.cbegin(), layerBasePrev), layerSplitDimensionBegin, nodeBboxBegin, IsNotLeaf{});
        thrust::scatter_if(layerBboxBegin, layerBboxEnd, thrust::next(node.rightChild.cbegin(), layerBasePrev), layerSplitDimensionBegin, nodeBboxBegin, IsNotLeaf{});

        splitNode<0>(layerBasePrev, x);
        splitNode<1>(layerBasePrev, y);
        splitNode<2>(layerBasePrev, z);

        node.splitDimension.resize(node.count, I(-1));
        node.splitPos.resize(node.count);
        node.leftChild.resize(node.count);
        node.rightChild.resize(node.count);
        node.polygonCountLeft.resize(node.count);
        node.polygonCountRight.resize(node.count);

        polygon.count += polygon.splittedCount;
    }

    node.parent.resize(node.count);
    thrust::scatter_if(thrust::make_counting_iterator<U>(0), thrust::make_counting_iterator<U>(node.count), node.leftChild.cbegin(), node.splitDimension.cbegin(), node.parent.begin(), IsNotLeaf{});
    thrust::scatter_if(thrust::make_counting_iterator<U>(0), thrust::make_counting_iterator<U>(node.count), node.rightChild.cbegin(), node.splitDimension.cbegin(), node.parent.begin(), IsNotLeaf{});

    assert(checkTree(x, y, z));

    populateLeafNodeTriangleRange();

    calculateRope<0>(Direction::kNegative, x, y, z);
    calculateRope<0>(Direction::kPositive, x, y, z);

    calculateRope<1>(Direction::kNegative, y, z, x);
    calculateRope<1>(Direction::kPositive, y, z, x);

    calculateRope<2>(Direction::kNegative, z, x, y);
    calculateRope<2>(Direction::kPositive, z, x, y);

    return tree;
}

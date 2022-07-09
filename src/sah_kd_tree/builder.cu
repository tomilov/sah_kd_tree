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
#include <thrust/tuple.h>

#include <cassert>

auto sah_kd_tree::Builder::operator()(const Params & sah) -> Tree
{
    if (triangleCount == 0) {
        return {};
    }

    x.calculateTriangleBbox(triangleCount);
    y.calculateTriangleBbox(triangleCount);
    z.calculateTriangleBbox(triangleCount);

    x.calculateRootNodeBbox();
    y.calculateRootNodeBbox();
    z.calculateRootNodeBbox();

    x.generateInitialEvent(triangleCount);
    y.generateInitialEvent(triangleCount);
    z.generateInitialEvent(triangleCount);

    U polygonCount = triangleCount;
    polygon.triangle.resize(polygonCount);
    thrust::sequence(polygon.triangle.begin(), polygon.triangle.end());
    polygon.node.assign(polygonCount, U(0));

    node.splitDimension.resize(1);
    node.splitPos.resize(1);
    node.leftChild.resize(1);
    node.rightChild.resize(1);
    node.polygonCount.assign(1, polygonCount);
    node.polygonCountLeft.resize(1);
    node.polygonCountRight.resize(1);

    // layer
    U layerBase = 0;
    U layerSize = 1;

    U leafNodeCount = 0;
    const auto isNotLeaf = [] __host__ __device__(I nodeSplitDimension) -> bool { return !(nodeSplitDimension < 0); };

    Tree tree;
    for (tree.depth = 0; tree.depth < sah.maxDepth; ++tree.depth) {
        filterLayerNodeOffset(layerBase, layerSize);

        x.findPerfectSplit(sah, layerSize, layerNodeOffset, node.polygonCount, y, z);
        y.findPerfectSplit(sah, layerSize, layerNodeOffset, node.polygonCount, z, x);
        z.findPerfectSplit(sah, layerSize, layerNodeOffset, node.polygonCount, x, y);

        selectNodeBestSplit(sah, layerBase, layerSize);

        auto layerSplitDimensionBegin = thrust::next(node.splitDimension.cbegin(), layerBase);
        auto layerSplitDimensionEnd = thrust::next(layerSplitDimensionBegin, layerSize);
        auto layerLeafNodeCount = U(thrust::count(layerSplitDimensionBegin, layerSplitDimensionEnd, I(-1)));
        leafNodeCount += layerLeafNodeCount;
        if (layerLeafNodeCount == layerSize) {
            break;
        }

        polygon.side.resize(polygonCount);
        polygon.eventRight.resize(polygonCount);

        x.template determinePolygonSide<0>(node.splitDimension, layerBase, polygon.eventRight, polygon.side);
        y.template determinePolygonSide<1>(node.splitDimension, layerBase, polygon.eventRight, polygon.side);
        z.template determinePolygonSide<2>(node.splitDimension, layerBase, polygon.eventRight, polygon.side);

        U splittedPolygonCount = getSplittedPolygonCount(layerBase, layerSize);

        {  // generate index for child node
            auto nodeLeftChildBegin = thrust::next(node.leftChild.begin(), layerBase);
            const auto toNodeCount = [] __host__ __device__(I layerSplitDimension) -> U { return (layerSplitDimension < 0) ? 0 : 2; };
            auto nodeLeftChildEnd = thrust::transform_exclusive_scan(layerSplitDimensionBegin, layerSplitDimensionEnd, nodeLeftChildBegin, toNodeCount, layerBase + layerSize, thrust::plus<U>{});

            auto nodeRightChildBegin = thrust::next(node.rightChild.begin(), layerBase);
            const auto toNodeRightChild = [] __host__ __device__(U nodeLeftChild) -> U { return nodeLeftChild + 1; };
            thrust::transform(nodeLeftChildBegin, nodeLeftChildEnd, nodeRightChildBegin, toNodeRightChild);
        }

        separateSplittedPolygon(layerBase, polygonCount, splittedPolygonCount);

        x.decoupleEventBoth(node.splitDimension, polygon.side);
        y.decoupleEventBoth(node.splitDimension, polygon.side);
        z.decoupleEventBoth(node.splitDimension, polygon.side);

        assert(polygon.side.size() == polygonCount);
        updatePolygonNode(layerBase);

        x.template splitPolygon<0>(node.splitDimension, node.splitPos, polygon.triangle, polygon.node, polygonCount, splittedPolygonCount, splittedPolygon, y, z);
        y.template splitPolygon<1>(node.splitDimension, node.splitPos, polygon.triangle, polygon.node, polygonCount, splittedPolygonCount, splittedPolygon, z, x);
        z.template splitPolygon<2>(node.splitDimension, node.splitPos, polygon.triangle, polygon.node, polygonCount, splittedPolygonCount, splittedPolygon, x, y);

        updateSplittedPolygonNode(polygonCount, splittedPolygonCount);

        x.mergeEvent(polygonCount, splittedPolygonCount, polygon.node, splittedPolygon);
        y.mergeEvent(polygonCount, splittedPolygonCount, polygon.node, splittedPolygon);
        z.mergeEvent(polygonCount, splittedPolygonCount, polygon.node, splittedPolygon);

        U layerBasePrev = layerBase;
        layerBase += layerSize;
        layerSize -= layerLeafNodeCount;
        layerSize += layerSize;

        node.polygonCount.resize(layerBase + layerSize);

        auto nodePolygonCountLeftBegin = thrust::next(node.polygonCountLeft.cbegin(), layerBasePrev);
        auto nodePolygonCountLeftEnd = thrust::next(node.polygonCountLeft.cbegin(), layerBase);
        thrust::scatter_if(nodePolygonCountLeftBegin, nodePolygonCountLeftEnd, thrust::next(node.leftChild.cbegin(), layerBasePrev), layerSplitDimensionBegin, node.polygonCount.begin(), isNotLeaf);
        auto nodePolygonCountRightBegin = thrust::next(node.polygonCountRight.cbegin(), layerBasePrev);
        auto nodePolygonCountRightEnd = thrust::next(node.polygonCountRight.cbegin(), layerBase);
        thrust::scatter_if(nodePolygonCountRightBegin, nodePolygonCountRightEnd, thrust::next(node.rightChild.cbegin(), layerBasePrev), layerSplitDimensionBegin, node.polygonCount.begin(), isNotLeaf);

        x.setNodeCount(layerBase + layerSize);
        y.setNodeCount(layerBase + layerSize);
        z.setNodeCount(layerBase + layerSize);

        auto nodeBboxBegin = thrust::make_zip_iterator(x.node.min.begin(), x.node.max.begin(), y.node.min.begin(), y.node.max.begin(), z.node.min.begin(), z.node.max.begin());
        auto layerBboxBegin = thrust::next(nodeBboxBegin, layerBasePrev);
        auto layerBboxEnd = thrust::next(nodeBboxBegin, layerBase);
        thrust::scatter_if(layerBboxBegin, layerBboxEnd, thrust::next(node.leftChild.cbegin(), layerBasePrev), layerSplitDimensionBegin, nodeBboxBegin, isNotLeaf);
        thrust::scatter_if(layerBboxBegin, layerBboxEnd, thrust::next(node.rightChild.cbegin(), layerBasePrev), layerSplitDimensionBegin, nodeBboxBegin, isNotLeaf);

        x.template splitNode<0>(layerBasePrev, layerBase, node.splitDimension, node.splitPos, node.leftChild, node.rightChild);
        y.template splitNode<1>(layerBasePrev, layerBase, node.splitDimension, node.splitPos, node.leftChild, node.rightChild);
        z.template splitNode<2>(layerBasePrev, layerBase, node.splitDimension, node.splitPos, node.leftChild, node.rightChild);

        node.splitDimension.resize(layerBase + layerSize, I(-1));
        node.splitPos.resize(layerBase + layerSize);
        node.leftChild.resize(layerBase + layerSize);
        node.rightChild.resize(layerBase + layerSize);
        node.polygonCountLeft.resize(layerBase + layerSize);
        node.polygonCountRight.resize(layerBase + layerSize);

        polygonCount += splittedPolygonCount;
    }

    populateLeafNodeTriangle(leafNodeCount);

    U nodeCount = layerBase + layerSize;
    node.parentNode.resize(nodeCount);

    auto nodeBegin = thrust::make_counting_iterator<U>(0);
    thrust::scatter_if(nodeBegin, thrust::next(nodeBegin, nodeCount), node.leftChild.cbegin(), node.splitDimension.cbegin(), node.parentNode.begin(), isNotLeaf);
    thrust::scatter_if(nodeBegin, thrust::next(nodeBegin, nodeCount), node.rightChild.cbegin(), node.splitDimension.cbegin(), node.parentNode.begin(), isNotLeaf);

    calculateRope<0>(nodeCount, node.rightChild, node.leftChild, y, z, x.node.leftRope);
    calculateRope<0>(nodeCount, node.leftChild, node.rightChild, y, z, x.node.rightRope);
    calculateRope<1>(nodeCount, node.rightChild, node.leftChild, z, x, y.node.leftRope);
    calculateRope<1>(nodeCount, node.leftChild, node.rightChild, z, x, y.node.rightRope);
    calculateRope<2>(nodeCount, node.rightChild, node.leftChild, x, y, z.node.leftRope);
    calculateRope<2>(nodeCount, node.leftChild, node.rightChild, x, y, z.node.rightRope);

    asm volatile("nop;");

    return tree;
}

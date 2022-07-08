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





#include <thrust/iterator/discard_iterator.h>
#include <cstdio>
#include <thrust/execution_policy.h>

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
    node.nodeLeft.resize(1);
    node.nodeRight.resize(1);
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
            auto nodeLeftBegin = thrust::next(node.nodeLeft.begin(), layerBase);
            const auto toNodeCount = [] __host__ __device__(I layerSplitDimension) -> U { return (layerSplitDimension < 0) ? 0 : 2; };
            auto nodeLeftEnd = thrust::transform_exclusive_scan(layerSplitDimensionBegin, layerSplitDimensionEnd, nodeLeftBegin, toNodeCount, layerBase + layerSize, thrust::plus<U>{});

            auto nodeRightBegin = thrust::next(node.nodeRight.begin(), layerBase);
            const auto toNodeRight = [] __host__ __device__(U nodeLeft) -> U { return nodeLeft + 1; };
            thrust::transform(nodeLeftBegin, nodeLeftEnd, nodeRightBegin, toNodeRight);
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
        thrust::scatter_if(nodePolygonCountLeftBegin, nodePolygonCountLeftEnd, thrust::next(node.nodeLeft.cbegin(), layerBasePrev), layerSplitDimensionBegin, node.polygonCount.begin(), isNotLeaf);
        auto nodePolygonCountRightBegin = thrust::next(node.polygonCountRight.cbegin(), layerBasePrev);
        auto nodePolygonCountRightEnd = thrust::next(node.polygonCountRight.cbegin(), layerBase);
        thrust::scatter_if(nodePolygonCountRightBegin, nodePolygonCountRightEnd, thrust::next(node.nodeRight.cbegin(), layerBasePrev), layerSplitDimensionBegin, node.polygonCount.begin(), isNotLeaf);

        x.setNodeCount(layerBase + layerSize);
        y.setNodeCount(layerBase + layerSize);
        z.setNodeCount(layerBase + layerSize);

        auto nodeBboxBegin = thrust::make_zip_iterator(x.node.min.begin(), x.node.max.begin(), y.node.min.begin(), y.node.max.begin(), z.node.min.begin(), z.node.max.begin());
        auto layerBboxBegin = thrust::next(nodeBboxBegin, layerBasePrev);
        auto layerBboxEnd = thrust::next(nodeBboxBegin, layerBase);
        thrust::scatter_if(layerBboxBegin, layerBboxEnd, thrust::next(node.nodeLeft.cbegin(), layerBasePrev), layerSplitDimensionBegin, nodeBboxBegin, isNotLeaf);
        thrust::scatter_if(layerBboxBegin, layerBboxEnd, thrust::next(node.nodeRight.cbegin(), layerBasePrev), layerSplitDimensionBegin, nodeBboxBegin, isNotLeaf);

        x.template splitNode<0>(layerBasePrev, layerBase, node.splitDimension, node.splitPos, node.nodeLeft, node.nodeRight);
        y.template splitNode<1>(layerBasePrev, layerBase, node.splitDimension, node.splitPos, node.nodeLeft, node.nodeRight);
        z.template splitNode<2>(layerBasePrev, layerBase, node.splitDimension, node.splitPos, node.nodeLeft, node.nodeRight);

        node.splitDimension.resize(layerBase + layerSize, I(-1));
        node.splitPos.resize(layerBase + layerSize);
        node.nodeLeft.resize(layerBase + layerSize);
        node.nodeRight.resize(layerBase + layerSize);
        node.polygonCountLeft.resize(layerBase + layerSize);
        node.polygonCountRight.resize(layerBase + layerSize);

        polygonCount += splittedPolygonCount;
    }

    populateLeafNodeTriangles(leafNodeCount);

    U nodeCount = layerBase + layerSize;
    node.parentNode.resize(nodeCount);

    auto nodeBegin = thrust::make_counting_iterator<U>(0);
    thrust::scatter_if(nodeBegin, thrust::next(nodeBegin, nodeCount), node.nodeLeft.cbegin(), node.splitDimension.cbegin(), node.parentNode.begin(), isNotLeaf);
    thrust::scatter_if(nodeBegin, thrust::next(nodeBegin, nodeCount), node.nodeRight.cbegin(), node.splitDimension.cbegin(), node.parentNode.begin(), isNotLeaf);

    auto yMins = y.node.min.data().get();
    auto yMaxs = y.node.max.data().get();
    auto zMins = z.node.min.data().get();
    auto zMaxs = z.node.max.data().get();
    auto parentNodes = node.parentNode.data().get();
    auto leftChildren = node.nodeLeft.data().get();
    auto rightChildren = node.nodeRight.data().get();
    auto splitDimensions = node.splitDimension.data().get();
    auto splitPositions = node.splitPos.data().get();
    constexpr I dimension = 0;
    const auto getRopeRight = [yMins, yMaxs, zMins, zMaxs, parentNodes, leftChildren, rightChildren, splitDimensions, splitPositions] __host__ __device__(U node) -> U
    {
        //assert(splitDimensions[node] < 0);
        U siblingNode = node;
        for (;;) {
            if (siblingNode == 0) {
                return 0;
            }
            U parentNode = parentNodes[siblingNode];
            if (splitDimensions[parentNode] == dimension) {
                if (siblingNode == leftChildren[parentNode]) {
                    if (siblingNode == node) {
                        return rightChildren[parentNode];
                    }
                    siblingNode = rightChildren[parentNode];
                    break;
                }
            }
            siblingNode = parentNode;
        }
        F yMin = yMins[node];
        F yMax = yMaxs[node];
        F zMin = zMins[node];
        F zMax = zMaxs[node];
        for (;;) {
            I siblingSplitDimension = splitDimensions[siblingNode];
            if (siblingSplitDimension < 0) {
                return siblingNode;
            } else if (siblingSplitDimension == dimension) {
                siblingNode = leftChildren[siblingNode];
            } else if (siblingSplitDimension == ((dimension + 1) % 3)) {
                F siblingSplitPosition = splitPositions[siblingNode];
                if (!(siblingSplitPosition < yMax)) {
                    siblingNode = leftChildren[siblingNode];
                } else if (!(yMin < siblingSplitPosition)) {
                    siblingNode = rightChildren[siblingNode];
                } else {
                    return siblingNode;
                }
            } else if (siblingSplitDimension == ((dimension + 2) % 3)) {
                F siblingSplitPosition = splitPositions[siblingNode];
                if (!(siblingSplitPosition < zMax)) {
                    siblingNode = leftChildren[siblingNode];
                } else if (!(zMin < siblingSplitPosition)) {
                    siblingNode = rightChildren[siblingNode];
                } else {
                    return siblingNode;
                }
            }
        }
        return node;
    };
    thrust::device_vector<U> ropeRight(nodeCount);
    //thrust::transform(node.leafNode.cbegin(), node.leafNode.cend(), ropeRight.begin(), getRopeRight);
    thrust::transform(thrust::make_counting_iterator<U>(0), thrust::make_counting_iterator<U>(nodeCount), ropeRight.begin(), getRopeRight);

    asm volatile("nop;");

    return tree;
}

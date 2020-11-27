#include "Utility.cuh"

#include <SahKdTree.hpp>

#include <thrust/advance.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/transform_scan.h>
#include <thrust/tuple.h>

#include <cassert>

auto SahKdTree::Builder::operator()(const Params & sah) -> Tree
{
    Timer timerTotal;
    Timer timer;

    auto triangleCount = U(x.triangle.a.size());
    assert(triangleCount == U(y.triangle.a.size()));
    assert(triangleCount == U(z.triangle.a.size()));

    x.calculateTriangleBbox();
    y.calculateTriangleBbox();
    z.calculateTriangleBbox();
    timer("calculateTriangleBbox");  // 9.330ms

    x.calculateRootNodeBbox();
    y.calculateRootNodeBbox();
    z.calculateRootNodeBbox();
    timer("calculateRootNodeBbox");  // 2.358ms

    x.generateInitialEvent();
    y.generateInitialEvent();
    z.generateInitialEvent();
    timer("generateInitialEvent");  // 141.887ms

    polygon.triangle.resize(triangleCount);
    thrust::sequence(polygon.triangle.begin(), polygon.triangle.end());
    polygon.node.assign(triangleCount, U(0));

    node.splitDimension.resize(1);
    node.splitPos.resize(1);
    node.nodeLeft.resize(1);
    node.nodeRight.resize(1);
    node.polygonCount.assign(1, triangleCount);
    node.polygonCountLeft.resize(1);
    node.polygonCountRight.resize(1);
    timer("init builder");  // 2.359ms

    // layer
    U layerBase = 0;
    U layerSize = 1;

    for (U depth = 0; depth < sah.maxDepth; ++depth) {
        thinLayerNodeOffset(layerBase, layerSize);
        timer("layerNodeOffset");  // 0.074ms

        x.findPerfectSplit(sah, layerSize, layerNodeOffset, y, z);
        y.findPerfectSplit(sah, layerSize, layerNodeOffset, z, x);
        z.findPerfectSplit(sah, layerSize, layerNodeOffset, x, y);
        timer("findPerfectSplit");  // 20.758ms

        selectNodeBestSplit(sah, layerBase, layerSize);
        timer("selectNodeBestSplit");  // 0.202ms

        auto layerSplitDimensionBegin = thrust::next(node.splitDimension.cbegin(), layerBase);
        auto layerSplitDimensionEnd = thrust::next(layerSplitDimensionBegin, layerSize);
        auto completedNodeCount = U(thrust::count(layerSplitDimensionBegin, layerSplitDimensionEnd, I(-1)));
        timer("completedNodeCount");  // 0.256ms
        if (completedNodeCount == layerSize) {
            break;
        }

        auto polygonCount = U(polygon.triangle.size());

        polygon.side.resize(polygonCount);
        polygon.eventLeft.resize(polygonCount);
        polygon.eventRight.resize(polygonCount);
        timer("resize polygon");  // 0.944ms

        x.determinePolygonSide(0, node.splitDimension, layerBase, polygon.eventLeft, polygon.eventRight, polygon.side);
        y.determinePolygonSide(1, node.splitDimension, layerBase, polygon.eventLeft, polygon.eventRight, polygon.side);
        z.determinePolygonSide(2, node.splitDimension, layerBase, polygon.eventLeft, polygon.eventRight, polygon.side);
        timer("determinePolygonSide");  // 7.020ms

        U splittedPolygonCount = getSplittedPolygonCount(layerBase, layerSize);
        timer("getSplittedPolygonCount");  // 0.048ms

        {  // generate index for child node
            auto nodeLeftBegin = thrust::next(node.nodeLeft.begin(), layerBase);
            auto toNodeCount = [] __host__ __device__(I layerSplitDimension) -> U { return (layerSplitDimension < 0) ? 0 : 2; };
            auto nodeLeftEnd = thrust::transform_exclusive_scan(layerSplitDimensionBegin, layerSplitDimensionEnd, nodeLeftBegin, toNodeCount, layerBase + layerSize, thrust::plus<U>{});

            auto nodeRightBegin = thrust::next(node.nodeRight.begin(), layerBase);
            auto toNodeRight = [] __host__ __device__(U nodeLeft) { return nodeLeft + 1; };
            thrust::transform(nodeLeftBegin, nodeLeftEnd, nodeRightBegin, toNodeRight);
        }
        timer("toNodePairIndices");  // 0.052ms

        separateSplittedPolygon(layerBase, polygonCount, splittedPolygonCount);
        timer("separateSplittedPolygon");  // 0.516ms

        x.decoupleEventBoth(node.splitDimension, polygon.side);
        y.decoupleEventBoth(node.splitDimension, polygon.side);
        z.decoupleEventBoth(node.splitDimension, polygon.side);
        timer("decoupleEventBoth");  // 7.316ms

        assert(polygon.side.size() == polygonCount);
        updatePolygonNode(layerBase);
        timer("updatePolygonNode");  // 0.727ms

        x.splitPolygon(0, node.splitDimension, node.splitPos, polygon.triangle, polygon.node, polygonCount, splittedPolygonCount, splittedPolygon, y, z);
        y.splitPolygon(1, node.splitDimension, node.splitPos, polygon.triangle, polygon.node, polygonCount, splittedPolygonCount, splittedPolygon, z, x);
        z.splitPolygon(2, node.splitDimension, node.splitPos, polygon.triangle, polygon.node, polygonCount, splittedPolygonCount, splittedPolygon, x, y);
        timer("splitPolygon");  // 0.006ms

        updateSplittedPolygonNode(polygonCount, splittedPolygonCount);
        timer("updateSplittedPolygonNode");  // 0.003ms

        x.mergeEvent(polygonCount, splittedPolygonCount, polygon.node, splittedPolygon);
        y.mergeEvent(polygonCount, splittedPolygonCount, polygon.node, splittedPolygon);
        z.mergeEvent(polygonCount, splittedPolygonCount, polygon.node, splittedPolygon);
        timer("mergeEvent");  // 44.897ms

        U layerBasePrev = layerBase;
        layerBase += layerSize;
        layerSize -= completedNodeCount;
        layerSize += layerSize;

        node.polygonCount.resize(layerBase + layerSize);

        auto isNotLeaf = [] __host__ __device__(I layerSplitDimension) -> bool { return !(layerSplitDimension < 0); };

        auto nodePolygonCountLeftBegin = thrust::next(node.polygonCountLeft.cbegin(), layerBasePrev);
        auto nodePolygonCountLeftEnd = thrust::next(node.polygonCountLeft.cbegin(), layerBase);
        thrust::scatter_if(nodePolygonCountLeftBegin, nodePolygonCountLeftEnd, thrust::next(node.nodeLeft.cbegin(), layerBasePrev), layerSplitDimensionBegin, node.polygonCount.begin(), isNotLeaf);
        auto nodePolygonCountRightBegin = thrust::next(node.polygonCountRight.cbegin(), layerBasePrev);
        auto nodePolygonCountRightEnd = thrust::next(node.polygonCountRight.cbegin(), layerBase);
        thrust::scatter_if(nodePolygonCountRightBegin, nodePolygonCountRightEnd, thrust::next(node.nodeRight.cbegin(), layerBasePrev), layerSplitDimensionBegin, node.polygonCount.begin(), isNotLeaf);
        timer("polygonCount");  // 0.056ms

        x.setNodeCount(layerBase + layerSize);
        y.setNodeCount(layerBase + layerSize);
        z.setNodeCount(layerBase + layerSize);
        timer("setNodeCount");  // 0.174ms

        auto nodeBboxBegin = thrust::make_zip_iterator(thrust::make_tuple(x.node.min.begin(), x.node.max.begin(), y.node.min.begin(), y.node.max.begin(), z.node.min.begin(), z.node.max.begin()));
        auto layerBboxBegin = thrust::next(nodeBboxBegin, layerBasePrev);
        auto layerBboxEnd = thrust::next(nodeBboxBegin, layerBase);
        thrust::scatter_if(layerBboxBegin, layerBboxEnd, thrust::next(node.nodeLeft.cbegin(), layerBasePrev), layerSplitDimensionBegin, nodeBboxBegin, isNotLeaf);
        thrust::scatter_if(layerBboxBegin, layerBboxEnd, thrust::next(node.nodeRight.cbegin(), layerBasePrev), layerSplitDimensionBegin, nodeBboxBegin, isNotLeaf);
        timer("setNodeBbox");  // 0.031ms

        x.splitNode(0, layerBasePrev, layerBase, node.splitDimension, node.splitPos, node.nodeLeft, node.nodeRight);
        y.splitNode(1, layerBasePrev, layerBase, node.splitDimension, node.splitPos, node.nodeLeft, node.nodeRight);
        z.splitNode(2, layerBasePrev, layerBase, node.splitDimension, node.splitPos, node.nodeLeft, node.nodeRight);
        timer("splitNode");  // 0.062ms

        node.splitDimension.resize(layerBase + layerSize, I(-1));
        node.splitPos.resize(layerBase + layerSize);
        node.nodeLeft.resize(layerBase + layerSize);
        node.nodeRight.resize(layerBase + layerSize);
        node.polygonCountLeft.resize(layerBase + layerSize);
        node.polygonCountRight.resize(layerBase + layerSize);
        timer("resizeNode");  // 0.168ms
        std::cout << depth << std::endl;
    }
    timerTotal("total");  // 236.149ms

    Tree tree;
    // calculate node parent
    // sort value (polygon) by key (polygon.node)
    // reduce value (counter, 1) by operation (project1st, plus) and key (node) to (key (node), value (offset, count))
    // scatter value (offset, count) to (node.nodeLeft, node.nodeRight) at key (node)
    return tree;
}

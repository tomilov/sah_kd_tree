#include "utility.cuh"

#include <sah_kd_tree/builder.hpp>

#include <thrust/advance.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/transform_scan.h>

#include <cassert>

auto SahKdTree::Builder::operator()(const Params & sah) -> SahKdTree
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
    U baseNode = 0;
    U nodeCount = 1;

    for (U depth = 0; depth < sah.maxDepth; ++depth) {
        {
            layerNodeOffset.resize(nodeCount);

            auto isNodeNotEmpty = [] __host__ __device__(U nodePolygonCount) -> bool { return nodePolygonCount != 0; };
            auto layerNodeBegin = thrust::make_counting_iterator<U>(0);
            auto layerNodeEnd = thrust::copy_if(layerNodeBegin, thrust::next(layerNodeBegin, nodeCount), thrust::next(node.polygonCount.cbegin(), baseNode), layerNodeOffset.begin(), isNodeNotEmpty);
            layerNodeOffset.erase(layerNodeEnd, layerNodeOffset.end());
            timer("layerNodeOffset");  // 0.074ms
        }

        x.findPerfectSplit(sah, nodeCount, layerNodeOffset, y, z);
        y.findPerfectSplit(sah, nodeCount, layerNodeOffset, z, x);
        z.findPerfectSplit(sah, nodeCount, layerNodeOffset, x, y);
        timer("findPerfectSplit");  // 20.758ms

        selectNodeBestSplit(sah, baseNode, nodeCount);
        timer("selectNodeBestSplit");  // 0.202ms

        auto nodeSplitDimensionBegin = thrust::next(node.splitDimension.cbegin(), baseNode);
        auto completedNodeCount = U(thrust::count(nodeSplitDimensionBegin, thrust::next(nodeSplitDimensionBegin, nodeCount), I(-1)));
        timer("completedNodeCount");  // 0.256ms
        if (completedNodeCount == nodeCount) {
            break;
        }

        auto polygonCount = U(polygon.triangle.size());

        polygon.side.resize(polygonCount);
        polygon.eventLeft.resize(polygonCount);
        polygon.eventRight.resize(polygonCount);
        timer("resize polygon");  // 0.944ms

        x.determinePolygonSide(0, node.splitDimension, baseNode, polygon.eventLeft, polygon.eventRight, polygon.side);
        y.determinePolygonSide(1, node.splitDimension, baseNode, polygon.eventLeft, polygon.eventRight, polygon.side);
        z.determinePolygonSide(2, node.splitDimension, baseNode, polygon.eventLeft, polygon.eventRight, polygon.side);
        timer("determinePolygonSide");  // 7.020ms

        U splittedPolygonCount = getSplittedPolygonCount(baseNode, nodeCount);
        timer("getSplittedPolygonCount");  // 0.048ms

        {  // generate index for child node
            auto nodeLeftBegin = thrust::next(node.nodeLeft.begin(), baseNode);
            auto toNodeCount = [] __host__ __device__(I nodeSplitDimension) -> U { return (nodeSplitDimension < 0) ? 0 : 2; };
            auto nodeLeftEnd = thrust::transform_exclusive_scan(nodeSplitDimensionBegin, thrust::next(nodeSplitDimensionBegin, nodeCount), nodeLeftBegin, toNodeCount, baseNode + nodeCount, thrust::plus<U>{});

            auto nodeRightBegin = thrust::next(node.nodeRight.begin(), baseNode);
            auto toNodeRight = [] __host__ __device__(U nodeLeft) { return nodeLeft + 1; };
            thrust::transform(nodeLeftBegin, nodeLeftEnd, nodeRightBegin, toNodeRight);
        }
        timer("toNodePairIndices");  // 0.052ms

        separateSplittedPolygon(baseNode, polygonCount, splittedPolygonCount);
        timer("separateSplittedPolygon");  // 0.516ms

        x.decoupleEventBoth(node.splitDimension, polygon.side);
        y.decoupleEventBoth(node.splitDimension, polygon.side);
        z.decoupleEventBoth(node.splitDimension, polygon.side);
        timer("decoupleEventBoth");  // 7.316ms

        updatePolygonNode(baseNode, polygonCount);
        timer("updatePolygonNode");  // 0.727ms

        x.splitPolygon(0, node.splitDimension, node.splitPos, polygon.triangle, polygon.node, polygonCount, splittedPolygonCount, splittedPolygon, y, z);
        y.splitPolygon(1, node.splitDimension, node.splitPos, polygon.triangle, polygon.node, polygonCount, splittedPolygonCount, splittedPolygon, z, x);
        z.splitPolygon(2, node.splitDimension, node.splitPos, polygon.triangle, polygon.node, polygonCount, splittedPolygonCount, splittedPolygon, x, y);
        timer("splitPolygon");  // 0.006ms

        updateSplittedPolygonNode(polygonCount, splittedPolygonCount);
        timer("updateSplittedPolygonNode");  // 0.003ms

        x.mergeEvent(polygonCount, polygon.node, splittedPolygonCount, splittedPolygon);
        y.mergeEvent(polygonCount, polygon.node, splittedPolygonCount, splittedPolygon);
        z.mergeEvent(polygonCount, polygon.node, splittedPolygonCount, splittedPolygon);
        timer("mergeEvent");  // 44.897ms
        break;
    }

    timerTotal("total");  // 231.211ms
    return {};
}

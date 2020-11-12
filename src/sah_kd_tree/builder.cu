#include "utility.cuh"

#include <sah_kd_tree/sah_kd_tree.hpp>

#include <thrust/advance.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
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

        U baseNodePrev = baseNode;
        baseNode += nodeCount;
        nodeCount -= completedNodeCount;
        nodeCount += nodeCount;

        node.polygonCount.resize(baseNode + nodeCount);

        auto isNotLeaf = [] __host__ __device__(I nodeSplitDimension) -> bool { return !(nodeSplitDimension < 0); };

        thrust::scatter_if(thrust::next(node.polygonCountLeft.cbegin(), baseNodePrev), thrust::next(node.polygonCountLeft.cbegin(), baseNode), thrust::next(node.nodeLeft.cbegin(), baseNodePrev), nodeSplitDimensionBegin, node.polygonCount.begin(),
                           isNotLeaf);
        thrust::scatter_if(thrust::next(node.polygonCountRight.cbegin(), baseNodePrev), thrust::next(node.polygonCountRight.cbegin(), baseNode), thrust::next(node.nodeRight.cbegin(), baseNodePrev), nodeSplitDimensionBegin, node.polygonCount.begin(),
                           isNotLeaf);
        timer("polygonCount");  // 0.056ms

        x.setNodeCount(baseNode + nodeCount);
        y.setNodeCount(baseNode + nodeCount);
        z.setNodeCount(baseNode + nodeCount);
        timer("setNodeCount");  // 0.174ms

        auto nodeBboxBegin = thrust::make_zip_iterator(thrust::make_tuple(x.node.min.begin(), x.node.max.begin(), y.node.min.begin(), y.node.max.begin(), z.node.min.begin(), z.node.max.begin()));
        auto layerBboxBegin = thrust::next(nodeBboxBegin, baseNodePrev);
        auto layerBboxEnd = thrust::next(nodeBboxBegin, baseNode);
        thrust::scatter_if(layerBboxBegin, layerBboxEnd, thrust::next(node.nodeLeft.cbegin(), baseNodePrev), nodeSplitDimensionBegin, nodeBboxBegin, isNotLeaf);
        thrust::scatter_if(layerBboxBegin, layerBboxEnd, thrust::next(node.nodeRight.cbegin(), baseNodePrev), nodeSplitDimensionBegin, nodeBboxBegin, isNotLeaf);
        timer("setNodeBbox");  // 0.031ms

        x.splitNode(0, baseNodePrev, baseNode, node.splitDimension, node.splitPos, node.nodeLeft, node.nodeRight);
        y.splitNode(1, baseNodePrev, baseNode, node.splitDimension, node.splitPos, node.nodeLeft, node.nodeRight);
        z.splitNode(2, baseNodePrev, baseNode, node.splitDimension, node.splitPos, node.nodeLeft, node.nodeRight);
        timer("splitNode");  // 0.062ms

        node.splitDimension.resize(baseNode + nodeCount, I(-1));
        node.splitPos.resize(baseNode + nodeCount);
        node.nodeLeft.resize(baseNode + nodeCount);
        node.nodeRight.resize(baseNode + nodeCount);
        node.polygonCountLeft.resize(baseNode + nodeCount);
        node.polygonCountRight.resize(baseNode + nodeCount);
        timer("resizeNode");  // 0.168ms
    }
    timerTotal("total");  // 236.149ms

    Tree tree;
    // calculate node parent
    // sort value (polygon) by key (polygon.node)
    // reduce value (counter, 1) by operation (project1st, plus) and key (node) to (key (node), value (offset, count))
    // scatter value (offset, count) to (node.nodeLeft, node.nodeRight) at key (node)
    return tree;
}

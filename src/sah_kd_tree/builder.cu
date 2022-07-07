#include <sah_kd_tree/sah_kd_tree.cuh>

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

    // layer
    U layerBase = 0;
    U layerSize = 1;

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
        auto completedNodeCount = U(thrust::count(layerSplitDimensionBegin, layerSplitDimensionEnd, I(-1)));
        if (completedNodeCount == layerSize) {
            break;
        }

        auto polygonCount = U(polygon.triangle.size());

        polygon.side.resize(polygonCount);
        polygon.eventRight.resize(polygonCount);

        x.determinePolygonSide(0, node.splitDimension, layerBase, polygon.eventRight, polygon.side);
        y.determinePolygonSide(1, node.splitDimension, layerBase, polygon.eventRight, polygon.side);
        z.determinePolygonSide(2, node.splitDimension, layerBase, polygon.eventRight, polygon.side);

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

        x.splitPolygon(0, node.splitDimension, node.splitPos, polygon.triangle, polygon.node, polygonCount, splittedPolygonCount, splittedPolygon, y, z);
        y.splitPolygon(1, node.splitDimension, node.splitPos, polygon.triangle, polygon.node, polygonCount, splittedPolygonCount, splittedPolygon, z, x);
        z.splitPolygon(2, node.splitDimension, node.splitPos, polygon.triangle, polygon.node, polygonCount, splittedPolygonCount, splittedPolygon, x, y);

        updateSplittedPolygonNode(polygonCount, splittedPolygonCount);

        x.mergeEvent(polygonCount, splittedPolygonCount, polygon.node, splittedPolygon);
        y.mergeEvent(polygonCount, splittedPolygonCount, polygon.node, splittedPolygon);
        z.mergeEvent(polygonCount, splittedPolygonCount, polygon.node, splittedPolygon);

        U layerBasePrev = layerBase;
        layerBase += layerSize;
        layerSize -= completedNodeCount;
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

        x.splitNode(0, layerBasePrev, layerBase, node.splitDimension, node.splitPos, node.nodeLeft, node.nodeRight);
        y.splitNode(1, layerBasePrev, layerBase, node.splitDimension, node.splitPos, node.nodeLeft, node.nodeRight);
        z.splitNode(2, layerBasePrev, layerBase, node.splitDimension, node.splitPos, node.nodeLeft, node.nodeRight);

        node.splitDimension.resize(layerBase + layerSize, I(-1));
        node.splitPos.resize(layerBase + layerSize);
        node.nodeLeft.resize(layerBase + layerSize);
        node.nodeRight.resize(layerBase + layerSize);
        node.polygonCountLeft.resize(layerBase + layerSize);
        node.polygonCountRight.resize(layerBase + layerSize);
    }

    // calculate node parent (needed for ropes calculation step)
    auto nodeCount = node.splitPos.size();
    node.parentNode.resize(nodeCount);
    auto nodeBegin = thrust::make_counting_iterator<U>(0);
    thrust::scatter_if(nodeBegin, thrust::next(nodeBegin, nodeCount), node.nodeLeft.cbegin(), node.splitDimension.cbegin(), node.parentNode.begin(), isNotLeaf);
    thrust::scatter_if(nodeBegin, thrust::next(nodeBegin, nodeCount), node.nodeRight.cbegin(), node.splitDimension.cbegin(), node.parentNode.begin(), isNotLeaf);
    // sort value (polygon) by key (polygon.node)
    // reduce value (counter, 1) by operation (project1st, plus) and key (node) to (key (node), value (offset, count))
    // scatter value (offset, count) to (node.nodeLeft, node.nodeRight) at key (node)
    return tree;
}

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

    const auto isNotLeaf = [] __host__ __device__(I nodeSplitDimension) -> bool { return !(nodeSplitDimension < 0); };

    Tree tree;
    for (tree.depth = 0; tree.depth < sah.maxDepth; ++tree.depth) {
        filterLayerNodeOffset();

        x.findPerfectSplit(sah, layer.size, layer.nodeOffset, node.polygonCount, y, z);
        y.findPerfectSplit(sah, layer.size, layer.nodeOffset, node.polygonCount, z, x);
        z.findPerfectSplit(sah, layer.size, layer.nodeOffset, node.polygonCount, x, y);

        selectNodeBestSplit(sah);

        auto layerSplitDimensionBegin = thrust::next(node.splitDimension.cbegin(), layer.base);
        auto layerSplitDimensionEnd = thrust::next(layerSplitDimensionBegin, layer.size);
        auto layerLeafNodeCount = U(thrust::count(layerSplitDimensionBegin, layerSplitDimensionEnd, I(-1)));
        node.leafCount += layerLeafNodeCount;
        if (layerLeafNodeCount == layer.size) {
            break;
        }

        polygon.side.resize(polygonCount);
        polygon.eventRight.resize(polygonCount);

        determinePolygonSide<0>(x);
        determinePolygonSide<1>(y);
        determinePolygonSide<2>(z);

        U splittedPolygonCount = getSplittedPolygonCount();

        {  // generate index for child node
            auto nodeLeftChildBegin = thrust::next(node.leftChild.begin(), layer.base);
            const auto toNodeCount = [] __host__ __device__(I layerSplitDimension) -> U { return (layerSplitDimension < 0) ? 0 : 2; };
            auto nodeLeftChildEnd = thrust::transform_exclusive_scan(layerSplitDimensionBegin, layerSplitDimensionEnd, nodeLeftChildBegin, toNodeCount, layer.base + layer.size, thrust::plus<U>{});

            auto nodeRightChildBegin = thrust::next(node.rightChild.begin(), layer.base);
            const auto toNodeRightChild = [] __host__ __device__(U nodeLeftChild) -> U { return nodeLeftChild + 1; };
            thrust::transform(nodeLeftChildBegin, nodeLeftChildEnd, nodeRightChildBegin, toNodeRightChild);
        }

        separateSplittedPolygon(polygonCount, splittedPolygonCount);

        x.decoupleEventBoth(node.splitDimension, polygon.side);
        y.decoupleEventBoth(node.splitDimension, polygon.side);
        z.decoupleEventBoth(node.splitDimension, polygon.side);

        assert(polygon.side.size() == polygonCount);
        updatePolygonNode();

        splitPolygon<0>(polygon.triangle, polygon.node, polygonCount, splittedPolygonCount, splittedPolygon, x, y, z);
        splitPolygon<1>(polygon.triangle, polygon.node, polygonCount, splittedPolygonCount, splittedPolygon, y, z, x);
        splitPolygon<2>(polygon.triangle, polygon.node, polygonCount, splittedPolygonCount, splittedPolygon, z, x, y);

        updateSplittedPolygonNode(polygonCount, splittedPolygonCount);

        x.mergeEvent(polygonCount, splittedPolygonCount, polygon.node, splittedPolygon);
        y.mergeEvent(polygonCount, splittedPolygonCount, polygon.node, splittedPolygon);
        z.mergeEvent(polygonCount, splittedPolygonCount, polygon.node, splittedPolygon);

        U layerBasePrev = layer.base;
        layer.base += layer.size;
        layer.size -= layerLeafNodeCount;
        layer.size += layer.size;

        node.polygonCount.resize(layer.base + layer.size);

        auto nodePolygonCountLeftBegin = thrust::next(node.polygonCountLeft.cbegin(), layerBasePrev);
        auto nodePolygonCountLeftEnd = thrust::next(node.polygonCountLeft.cbegin(), layer.base);
        thrust::scatter_if(nodePolygonCountLeftBegin, nodePolygonCountLeftEnd, thrust::next(node.leftChild.cbegin(), layerBasePrev), layerSplitDimensionBegin, node.polygonCount.begin(), isNotLeaf);
        auto nodePolygonCountRightBegin = thrust::next(node.polygonCountRight.cbegin(), layerBasePrev);
        auto nodePolygonCountRightEnd = thrust::next(node.polygonCountRight.cbegin(), layer.base);
        thrust::scatter_if(nodePolygonCountRightBegin, nodePolygonCountRightEnd, thrust::next(node.rightChild.cbegin(), layerBasePrev), layerSplitDimensionBegin, node.polygonCount.begin(), isNotLeaf);

        x.setNodeCount(layer.base + layer.size);
        y.setNodeCount(layer.base + layer.size);
        z.setNodeCount(layer.base + layer.size);

        auto nodeBboxBegin = thrust::make_zip_iterator(x.node.min.begin(), x.node.max.begin(), y.node.min.begin(), y.node.max.begin(), z.node.min.begin(), z.node.max.begin());
        auto layerBboxBegin = thrust::next(nodeBboxBegin, layerBasePrev);
        auto layerBboxEnd = thrust::next(nodeBboxBegin, layer.base);
        thrust::scatter_if(layerBboxBegin, layerBboxEnd, thrust::next(node.leftChild.cbegin(), layerBasePrev), layerSplitDimensionBegin, nodeBboxBegin, isNotLeaf);
        thrust::scatter_if(layerBboxBegin, layerBboxEnd, thrust::next(node.rightChild.cbegin(), layerBasePrev), layerSplitDimensionBegin, nodeBboxBegin, isNotLeaf);

        x.template splitNode<0>(layerBasePrev, layer.base, node.splitDimension, node.splitPos, node.leftChild, node.rightChild);
        y.template splitNode<1>(layerBasePrev, layer.base, node.splitDimension, node.splitPos, node.leftChild, node.rightChild);
        z.template splitNode<2>(layerBasePrev, layer.base, node.splitDimension, node.splitPos, node.leftChild, node.rightChild);

        node.splitDimension.resize(layer.base + layer.size, I(-1));
        node.splitPos.resize(layer.base + layer.size);
        node.leftChild.resize(layer.base + layer.size);
        node.rightChild.resize(layer.base + layer.size);
        node.polygonCountLeft.resize(layer.base + layer.size);
        node.polygonCountRight.resize(layer.base + layer.size);

        polygonCount += splittedPolygonCount;
    }

    U nodeCount = layer.base + layer.size;

    node.parentNode.resize(nodeCount);
    thrust::scatter_if(thrust::make_counting_iterator<U>(0), thrust::make_counting_iterator<U>(nodeCount), node.leftChild.cbegin(), node.splitDimension.cbegin(), node.parentNode.begin(), isNotLeaf);
    thrust::scatter_if(thrust::make_counting_iterator<U>(0), thrust::make_counting_iterator<U>(nodeCount), node.rightChild.cbegin(), node.splitDimension.cbegin(), node.parentNode.begin(), isNotLeaf);

    assert(checkTree(triangleCount, polygonCount, nodeCount));

    populateLeafNodeTriangleRange();

    calculateRope<0>(nodeCount, true, y, z, x.node.leftRope);
    calculateRope<0>(nodeCount, false, y, z, x.node.rightRope);
    calculateRope<1>(nodeCount, true, z, x, y.node.leftRope);
    calculateRope<1>(nodeCount, false, z, x, y.node.rightRope);
    calculateRope<2>(nodeCount, true, x, y, z.node.leftRope);
    calculateRope<2>(nodeCount, false, x, y, z.node.rightRope);

    return tree;
}

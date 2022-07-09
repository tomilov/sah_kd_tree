#pragma once

#include <sah_kd_tree/sah_kd_tree_export.h>
#include <sah_kd_tree/types.cuh>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#include <limits>

namespace sah_kd_tree
{
struct SAH_KD_TREE_EXPORT Params
{
    F emptinessFactor = 0.8f;   // (0, 1]
    F traversalCost = 2.0f;     // (0, inf)
    F intersectionCost = 1.0f;  // (0, inf)
    U maxDepth = std::numeric_limits<U>::max();
};

struct SAH_KD_TREE_EXPORT Tree
{
    // TODO(tomilov):
    U depth = std::numeric_limits<U>::max();
};

struct SAH_KD_TREE_EXPORT Projection
{
    struct Triangle
    {
        thrust::device_ptr<const F> a, b, c;
    } triangle;

    struct Polygon
    {
        thrust::device_vector<F> min, max;
    } polygon;

    struct Node
    {
        thrust::device_vector<F> min, max;
    } node;

    struct Event
    {
        thrust::device_vector<U> node;
        thrust::device_vector<F> pos;
        thrust::device_vector<I> kind;
        thrust::device_vector<U> polygon;

        thrust::device_vector<U> polygonCountLeft, polygonCountRight;  // or eventLeft, eventRight mutually exclusive
    } event;

    struct Layer
    {
        thrust::device_vector<F> splitCost;
        thrust::device_vector<U> splitEvent;
        thrust::device_vector<F> splitPos;

        thrust::device_vector<U> polygonCountLeft, polygonCountRight;
    } layer;

    void calculateTriangleBbox(U triangleCount);
    void calculateRootNodeBbox();
    void generateInitialEvent(U triangleCount);

    void findPerfectSplit(const Params & sah, U layerSize, const thrust::device_vector<U> & layerNodeOffset, const thrust::device_vector<U> & nodePolygonCount, const Projection & y, const Projection & z);
    template<I dimension>
    void determinePolygonSide(const thrust::device_vector<I> & nodeSplitDimension, U layerBase, thrust::device_vector<U> & polygonEventRight, thrust::device_vector<I> & polygonSide);
    void decoupleEventBoth(const thrust::device_vector<I> & nodeSplitDimension, const thrust::device_vector<I> & polygonSide);
    template<I dimension>
    void splitPolygon(const thrust::device_vector<I> & nodeSplitDimension, const thrust::device_vector<F> & nodeSplitPos, const thrust::device_vector<U> & polygonTriangle, const thrust::device_vector<U> & polygonNode, U polygonCount,
                      U splittedPolygonCount, const thrust::device_vector<U> & splittedPolygon, const Projection & y, const Projection & z);

    void mergeEvent(U polygonCount, U splittedPolygonCount, const thrust::device_vector<U> & polygonNode, const thrust::device_vector<U> & splittedPolygon);
    void setNodeCount(U layerSize);
    template<I dimension>
    void splitNode(U layerBasePrev, U layerBase, const thrust::device_vector<I> & nodeSplitDimension, const thrust::device_vector<F> & nodeSplitPos, const thrust::device_vector<U> & leftChild, const thrust::device_vector<U> & rightChild);
};

extern template void Projection::determinePolygonSide<0>(const thrust::device_vector<I> & nodeSplitDimension, U layerBase, thrust::device_vector<U> & polygonEventRight, thrust::device_vector<I> & polygonSide) SAH_KD_TREE_EXPORT;
extern template void Projection::determinePolygonSide<1>(const thrust::device_vector<I> & nodeSplitDimension, U layerBase, thrust::device_vector<U> & polygonEventRight, thrust::device_vector<I> & polygonSide) SAH_KD_TREE_EXPORT;
extern template void Projection::determinePolygonSide<2>(const thrust::device_vector<I> & nodeSplitDimension, U layerBase, thrust::device_vector<U> & polygonEventRight, thrust::device_vector<I> & polygonSide) SAH_KD_TREE_EXPORT;

extern template void Projection::splitPolygon<0>(const thrust::device_vector<I> & nodeSplitDimension, const thrust::device_vector<F> & nodeSplitPos, const thrust::device_vector<U> & polygonTriangle, const thrust::device_vector<U> & polygonNode,
                                                 U polygonCount, U splittedPolygonCount, const thrust::device_vector<U> & splittedPolygon, const Projection & y, const Projection & z) SAH_KD_TREE_EXPORT;
extern template void Projection::splitPolygon<1>(const thrust::device_vector<I> & nodeSplitDimension, const thrust::device_vector<F> & nodeSplitPos, const thrust::device_vector<U> & polygonTriangle, const thrust::device_vector<U> & polygonNode,
                                                 U polygonCount, U splittedPolygonCount, const thrust::device_vector<U> & splittedPolygon, const Projection & y, const Projection & z) SAH_KD_TREE_EXPORT;
extern template void Projection::splitPolygon<2>(const thrust::device_vector<I> & nodeSplitDimension, const thrust::device_vector<F> & nodeSplitPos, const thrust::device_vector<U> & polygonTriangle, const thrust::device_vector<U> & polygonNode,
                                                 U polygonCount, U splittedPolygonCount, const thrust::device_vector<U> & splittedPolygon, const Projection & y, const Projection & z) SAH_KD_TREE_EXPORT;

extern template void Projection::splitNode<0>(U layerBasePrev, U layerBase, const thrust::device_vector<I> & nodeSplitDimension, const thrust::device_vector<F> & nodeSplitPos, const thrust::device_vector<U> & leftChild,
                                              const thrust::device_vector<U> & rightChild) SAH_KD_TREE_EXPORT;
extern template void Projection::splitNode<1>(U layerBasePrev, U layerBase, const thrust::device_vector<I> & nodeSplitDimension, const thrust::device_vector<F> & nodeSplitPos, const thrust::device_vector<U> & leftChild,
                                              const thrust::device_vector<U> & rightChild) SAH_KD_TREE_EXPORT;
extern template void Projection::splitNode<2>(U layerBasePrev, U layerBase, const thrust::device_vector<I> & nodeSplitDimension, const thrust::device_vector<F> & nodeSplitPos, const thrust::device_vector<U> & leftChild,
                                              const thrust::device_vector<U> & rightChild) SAH_KD_TREE_EXPORT;

struct SAH_KD_TREE_EXPORT Builder
{
    U triangleCount = 0;

    Projection x, y, z;

    struct Polygon
    {
        thrust::device_vector<U> triangle;
        thrust::device_vector<U> node;
        thrust::device_vector<I> side;
        thrust::device_vector<U> eventRight;  // right event in diverse best dimensions
    } polygon;

    struct Node
    {
        thrust::device_vector<I> splitDimension;
        thrust::device_vector<F> splitPos;                                           // TODO: splitDimension can be packed into 2 lsb of splitPos
        thrust::device_vector<U> leftChild, rightChild;                              // left child node and right child node if not leaf, polygon range otherwise
        thrust::device_vector<U> polygonCount, polygonCountLeft, polygonCountRight;  // unique polygon count in the current node, in its left child node and in its right child node correspondingly
        thrust::device_vector<U> parentNode;                                         // temporarily needed to build  ropes
        thrust::device_vector<U> leafNode;
    } node;  // TODO: optimize out node.rightChild

    thrust::device_vector<U> layerNodeOffset;

    thrust::device_vector<U> splittedPolygon;

    void filterLayerNodeOffset(U layerBase, U layerSize);
    void selectNodeBestSplit(const Params & sah, U layerBase, U layerSize);
    U getSplittedPolygonCount(U layerBase, U layerSize);
    void separateSplittedPolygon(U layerBase, U polygonCount, U splittedPolygonCount);
    void updatePolygonNode(U layerBase);
    void updateSplittedPolygonNode(U polygonCount, U splittedPolygonCount);
    void populateLeafNodeTriangles(U leafNodeCount);

    Tree operator()(const Params & sah);
};

}  // namespace sah_kd_tree

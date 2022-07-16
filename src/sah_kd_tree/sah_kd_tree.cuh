#pragma once

#include <sah_kd_tree/sah_kd_tree_export.h>
#include <sah_kd_tree/types.cuh>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/tuple.h>

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
    struct Doubler
    {
        __host__ __device__ thrust::tuple<U, U> operator()(U value) const
        {
            return {value, value};
        }
    } doubler;

    struct ToEventPos
    {
        __host__ __device__ F operator()(I eventKind, thrust::tuple<F, F> bbox) const
        {
            return (eventKind < 0) ? thrust::get<1>(bbox) : thrust::get<0>(bbox);
        }
    } toEventPos;

    struct Triangle
    {
        U count = 0;
        thrust::device_ptr<const F> a, b, c;
    } triangle;

    struct Polygon
    {
        thrust::device_vector<F> min, max;
    } polygon;

    struct Node
    {
        thrust::device_vector<F> min, max;
        thrust::device_vector<U> leftRope, rightRope;
    } node;

    struct Event
    {
        U count = 0;
        thrust::device_vector<U> node;
        thrust::device_vector<F> pos;
        thrust::device_vector<I> kind;  // scale event kind by polygon value
        thrust::device_vector<U> polygon;

        thrust::device_vector<U> polygonCountLeft, polygonCountRight;  // or eventLeft, eventRight mutually exclusive
    } event;

    struct Layer
    {
        thrust::device_vector<F> splitCost;
        thrust::device_vector<U> splitEvent;
        thrust::device_vector<F> splitPos;

        thrust::device_vector<U> polygonCountLeft, polygonCountRight;
        thrust::device_vector<U> splittedPolygonCount;  // can be optimized out
    } layer;

    void calculateTriangleBbox();
    void calculateRootNodeBbox();
    void generateInitialEvent();

    void findPerfectSplit(const Params & sah, U layerSize, const thrust::device_vector<U> & layerNodeOffset, const thrust::device_vector<U> & nodePolygonCount, const Projection & y, const Projection & z);
    void decoupleEventBoth(const thrust::device_vector<I> & nodeSplitDimension, const thrust::device_vector<I> & polygonSide);

    void mergeEvent(U polygonCount, U splittedPolygonCount, const thrust::device_vector<U> & polygonNode, const thrust::device_vector<U> & splittedPolygon);
};

struct SAH_KD_TREE_EXPORT Builder
{
    struct IsNotLeaf
    {
        __host__ __device__ bool operator()(I nodeSplitDimension) const
        {
            return !(nodeSplitDimension < 0);
        }
    } isNotLeaf;

    struct IsNodeNotEmpty
    {
        __host__ __device__ bool operator()(U nodePolygonCount) const
        {
            return nodePolygonCount != 0;
        }
    } isNodeNotEmpty;

    struct Polygon
    {
        U count = 0;
        U splittedCount = 0;

        thrust::device_vector<U> triangle;
        thrust::device_vector<U> node;
        thrust::device_vector<I> side;
        thrust::device_vector<U> eventRight;  // right event in diverse best dimensions
    } polygon;

    struct Node
    {
        U count = 1;  // always equal layer.base + layer.size

        thrust::device_vector<I> splitDimension;
        thrust::device_vector<F> splitPos;                                           // TODO: splitDimension can be packed into 2 lsb of splitPos
        thrust::device_vector<U> leftChild, rightChild;                              // left child node and right child node if not leaf, polygon range otherwise
        thrust::device_vector<U> polygonCount, polygonCountLeft, polygonCountRight;  // unique polygon count in the current node, in its left child node and in its right child node correspondingly
        thrust::device_vector<U> parent;                                             // temporarily needed to build  ropes
    } node;  // TODO: optimize out node.rightChild

    struct Leaf
    {
        U count = 0;

        thrust::device_vector<U> node;
    } leaf;

    struct Layer
    {
        U base = 0;
        U size = 1;

        thrust::device_vector<U> nodeOffset;
    } layer;

    thrust::device_vector<U> splittedPolygon;

    void filterLayerNodeOffset();
    void selectNodeBestSplit(const Params & sah, const Projection & x, const Projection & y, const Projection & z);
    template<I dimension>
    void determinePolygonSide(const Projection & projection);
    void updateSplittedPolygonCount();
    void separateSplittedPolygon();
    void updatePolygonNode();
    template<I dimension>
    void splitPolygon(Projection & x, const Projection & y, const Projection & z) const;
    void updateSplittedPolygonNode();
    void setNodeCount(Projection & x, Projection & y, Projection & z) const;
    template<I dimension>
    void splitNode(U layerBasePrev, Projection & projection) const;
    void resizeNode();
    void populateNodeParent();
    void populateLeafNodeTriangleRange();

    bool checkTree(const Projection & x, const Projection & y, const Projection & z) const;

    template<I dimension, bool forth>
    void calculateRope(Projection & x, const Projection & y, const Projection & z) const;

    Tree operator()(const Params & sah, Projection & x, Projection & y, Projection & z);
};

extern template void Builder::determinePolygonSide<0>(const Projection & x) SAH_KD_TREE_EXPORT;
extern template void Builder::determinePolygonSide<1>(const Projection & y) SAH_KD_TREE_EXPORT;
extern template void Builder::determinePolygonSide<2>(const Projection & z) SAH_KD_TREE_EXPORT;

extern template void Builder::splitPolygon<0>(Projection & x, const Projection & y, const Projection & z) const SAH_KD_TREE_EXPORT;
extern template void Builder::splitPolygon<1>(Projection & y, const Projection & z, const Projection & x) const SAH_KD_TREE_EXPORT;
extern template void Builder::splitPolygon<2>(Projection & z, const Projection & x, const Projection & y) const SAH_KD_TREE_EXPORT;

extern template void Builder::splitNode<0>(U layerBasePrev, Projection & x) const SAH_KD_TREE_EXPORT;
extern template void Builder::splitNode<1>(U layerBasePrev, Projection & y) const SAH_KD_TREE_EXPORT;
extern template void Builder::splitNode<2>(U layerBasePrev, Projection & z) const SAH_KD_TREE_EXPORT;

extern template void Builder::calculateRope<0, false>(Projection & x, const Projection & y, const Projection & z) const SAH_KD_TREE_EXPORT;
extern template void Builder::calculateRope<0, true>(Projection & x, const Projection & y, const Projection & z) const SAH_KD_TREE_EXPORT;
extern template void Builder::calculateRope<1, false>(Projection & y, const Projection & z, const Projection & x) const SAH_KD_TREE_EXPORT;
extern template void Builder::calculateRope<1, true>(Projection & y, const Projection & z, const Projection & x) const SAH_KD_TREE_EXPORT;
extern template void Builder::calculateRope<2, false>(Projection & z, const Projection & x, const Projection & y) const SAH_KD_TREE_EXPORT;
extern template void Builder::calculateRope<2, true>(Projection & z, const Projection & x, const Projection & y) const SAH_KD_TREE_EXPORT;
}  // namespace sah_kd_tree

#pragma once

#include <SahKdTree/Types.hpp>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#include <limits>

namespace SahKdTree
{
struct Params
{
    F emptinessFactor = 0.8f;   // (0, 1]
    F traversalCost = 2.0f;     // (0, inf)
    F intersectionCost = 1.0f;  // (0, inf)
    U maxDepth = std::numeric_limits<U>::max();
};

struct Tree
{
    // TODO(tomilov):
    U depth = std::numeric_limits<U>::max();
};

struct Projection
{
    struct
    {
        thrust::device_vector<F> a, b, c;
    } triangle;

    struct
    {
        thrust::device_vector<F> min, max;
    } polygon;

    struct
    {
        thrust::device_vector<F> min, max;
    } node;

    struct
    {
        thrust::device_vector<U> node;
        thrust::device_vector<F> pos;
        thrust::device_vector<I> kind;
        thrust::device_vector<U> polygon;

        thrust::device_vector<U> polygonCountLeft, polygonCountRight;  // or eventLeft, eventRight mutually exclusive
    } event;

    struct
    {
        thrust::device_vector<F> splitCost;
        thrust::device_vector<U> splitEvent;
        thrust::device_vector<F> splitPos;

        thrust::device_vector<U> polygonCountLeft, polygonCountRight;
    } layer;

    void calculateTriangleBbox();
    void calculateRootNodeBbox();
    void generateInitialEvent();

    void findPerfectSplit(const Params & sah, U layerSize, const thrust::device_vector<U> & layerNodeOffset, const Projection & y, const Projection & z);
    void determinePolygonSide(I dimension, const thrust::device_vector<I> & nodeSplitDimension, U layerBase, thrust::device_vector<U> & polygonEventLeft, thrust::device_vector<U> & polygonEventRight, thrust::device_vector<I> & polygonSide);
    void decoupleEventBoth(const thrust::device_vector<I> & nodeSplitDimension, const thrust::device_vector<I> & polygonSide);
    void splitPolygon(I dimension, const thrust::device_vector<I> & nodeSplitDimension, const thrust::device_vector<F> & nodeSplitPos, const thrust::device_vector<U> & polygonTriangle, const thrust::device_vector<U> & polygonNode, U polygonCount,
                      U splittedPolygonCount, const thrust::device_vector<U> & splittedPolygon, const Projection & y, const Projection & z);

    void mergeEvent(U polygonCount, U splittedPolygonCount, const thrust::device_vector<U> & polygonNode, const thrust::device_vector<U> & splittedPolygon);
    void setNodeCount(U layerSize);
    void splitNode(I dimension, U layerBasePrev, U layerBase, const thrust::device_vector<I> & nodeSplitDimension, const thrust::device_vector<F> & nodeSplitPos, const thrust::device_vector<U> & nodeLeft, const thrust::device_vector<U> & nodeRight);
};

struct Builder
{
    Projection x, y, z;

    struct
    {
        thrust::device_vector<U> triangle;
        thrust::device_vector<U> node;
        thrust::device_vector<I> side;
        thrust::device_vector<U> eventLeft, eventRight;  // corresponding left and right event in diverse best dimensions
    } polygon;

    struct
    {
        thrust::device_vector<I> splitDimension;
        thrust::device_vector<F> splitPos;
        thrust::device_vector<U> nodeLeft, nodeRight;                                // left child node and right child node if not leaf, polygon range otherwise
        thrust::device_vector<U> polygonCount, polygonCountLeft, polygonCountRight;  // unique polygon count in the current node, in its left child node and in its right child node correspondingly
    } node;                                                                          // TODO: optimize out node.nodeLeft

    thrust::device_vector<U> layerNodeOffset;

    thrust::device_vector<U> splittedPolygon;

    void setTriangle(thrust::device_ptr<const Triangle> triangleBegin, thrust::device_ptr<const Triangle> triangleEnd);
    void thinLayerNodeOffset(U layerBase, U layerSize);
    void selectNodeBestSplit(const Params & sah, U layerBase, U layerSize);
    U getSplittedPolygonCount(U layerBase, U layerSize);
    void separateSplittedPolygon(U layerBase, U polygonCount, U splittedPolygonCount);
    void updatePolygonNode(U layerBase);
    void updateSplittedPolygonNode(U polygonCount, U splittedPolygonCount);

    Tree operator()(const Params & sah);
};
}  // namespace SahKdTree

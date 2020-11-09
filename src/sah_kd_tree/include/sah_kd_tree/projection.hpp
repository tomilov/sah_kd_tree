#pragma once

#include <sah_kd_tree/config.hpp>

#include <thrust/device_vector.h>

namespace SahKdTree
{
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

        thrust::device_vector<U> countLeft, countRight;
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

    void findPerfectSplit(const Params & sah, U nodeCount, const thrust::device_vector<U> & layerNodeOffset, const Projection & y, const Projection & z);
    void determinePolygonSide(I dimension, const thrust::device_vector<I> & nodeSplitDimension, U baseNode, thrust::device_vector<U> & polygonEventLeft, thrust::device_vector<U> & polygonEventRight, thrust::device_vector<I> & polygonSide);
    void decoupleEventBoth(const thrust::device_vector<I> & nodeSplitDimension, const thrust::device_vector<I> & polygonSide);
    void splitPolygon(I dimension, const thrust::device_vector<I> & nodeSplitDimension, const thrust::device_vector<F> & nodeSplitPos, const thrust::device_vector<U> & polygonTriangle, const thrust::device_vector<U> & polygonNode, U polygonCount,
                      U splittedPolygonCount, const thrust::device_vector<U> & splittedPolygon, const Projection & y, const Projection & z);

    void mergeEvent(U polygonCount, const thrust::device_vector<U> & polygonNode, U splittedPolygonCount, const thrust::device_vector<U> & splittedPolygon);
    void setNodeCount(U nodeCount);
    void splitNode(I dimension, U baseNodePrev, U baseNode, const thrust::device_vector<I> & nodeSplitDimension, const thrust::device_vector<F> & nodeSplitPos, const thrust::device_vector<U> & nodeLeft, const thrust::device_vector<U> & nodeRight);
};
}  // namespace SahKdTree

#pragma once

#include <sah_kd_tree/config.hpp>
#include <sah_kd_tree/projection.hpp>
#include <sah_kd_tree/sah_kd_tree.hpp>
#include <sah_kd_tree/types.hpp>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

namespace SahKdTree
{
struct Builder
{
    Projection x, y, z;

    struct
    {
        thrust::device_vector<U> triangle;
        thrust::device_vector<U> node;
        thrust::device_vector<I> side;
        thrust::device_vector<U> leftEvent, rightEvent;  // corresponding left and right event in diverse best dimensions
    } polygon;

    struct
    {
        thrust::device_vector<I> splitDimension;
        thrust::device_vector<F> splitPos;
        thrust::device_vector<U> leftNode, rightNode;                                // left child node and right child node if not leaf, polygon range otherwise
        thrust::device_vector<U> polygonCount, polygonCountLeft, polygonCountRight;  // unique polygon count in the current node, in its left child node and in its right child node correspondingly
    } node;                                                                          // TODO: optimize out node.leftNode

    thrust::device_vector<U> layerNodeOffset;

    void setTriangle(thrust::device_ptr<const Triangle> triangleBegin, thrust::device_ptr<const Triangle> triangleEnd);
    void selectNodeBestSplit(const Params & sah, U baseNode, U nodeCount);
    U getSplittedPolygonCount(U baseNode, U nodeCount);

    SahKdTree operator()(const Params & sah);
};
}  // namespace SahKdTree

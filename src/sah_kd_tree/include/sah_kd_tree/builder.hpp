#pragma once

#include <sah_kd_tree/config.hpp>
#include <sah_kd_tree/projection.hpp>
#include <sah_kd_tree/tree.hpp>
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
    void selectNodeBestSplit(const Params & sah, U baseNode, U nodeCount);
    U getSplittedPolygonCount(U baseNode, U nodeCount);
    void separateSplittedPolygon(U baseNode, U polygonCount, U splittedPolygonCount);
    void updatePolygonNode(U baseNode, U polygonCount);
    void updateSplittedPolygonNode(U polygonCount, U splittedPolygonCount);

    Tree operator()(const Params & sah);
};
}  // namespace SahKdTree

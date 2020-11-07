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
    Projection<0> x;
    Projection<1> y;
    Projection<2> z;

    struct
    {
        thrust::device_vector<U> triangle;
        thrust::device_vector<U> node;
    } polygon;

    struct
    {
        thrust::device_vector<I> splitDimension;
        thrust::device_vector<F> splitPos;
        thrust::device_vector<U> l, r;  // left child node and right child node if not leaf, polygon range otherwise
        struct
        {
            thrust::device_vector<U> node, l, r;  // unique polygon count in current node, in its left child node and in its right child node correspondingly
        } polygonCount;
    } node;  // TODO: optimize out node.l

    void setTriangle(thrust::device_ptr<const Triangle> triangleBegin, thrust::device_ptr<const Triangle> triangleEnd);
    SahKdTree operator()(const Params & sah);
};
}  // namespace SahKdTree

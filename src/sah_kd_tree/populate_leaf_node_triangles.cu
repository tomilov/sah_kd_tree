#include <sah_kd_tree/sah_kd_tree.cuh>

#include <thrust/advance.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/sort.h>

void sah_kd_tree::Builder::populateLeafNodeTriangles(U leafNodeCount)
{
    thrust::sort_by_key(polygon.node.begin(), polygon.node.end(), polygon.triangle.begin());

    node.leafNode.resize(leafNodeCount);
    thrust::device_vector<U> leafPolygonCount(leafNodeCount);
    auto leafPolygonCountEnd = thrust::reduce_by_key(polygon.node.begin(), polygon.node.end(), thrust::make_constant_iterator<U>(1), node.leafNode.begin(), leafPolygonCount.begin());
    assert(leafPolygonCountEnd.first == node.leafNode.end());

    thrust::device_vector<U> leafPolygonOffset(leafNodeCount);
    thrust::exclusive_scan(leafPolygonCount.cbegin(), leafPolygonCount.cend(), leafPolygonOffset.begin());

    auto leafPolygonBegin = thrust::make_zip_iterator(leafPolygonOffset.cbegin(), leafPolygonCount.cbegin());
    auto leafPolygonOutputBegin = thrust::make_zip_iterator(node.leftChild.begin(), node.rightChild.begin());
    thrust::scatter(leafPolygonBegin, thrust::next(leafPolygonBegin, leafNodeCount), node.leafNode.cbegin(), leafPolygonOutputBegin);
}

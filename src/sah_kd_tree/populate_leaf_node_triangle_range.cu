#include <sah_kd_tree/sah_kd_tree.cuh>

#include <thrust/device_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/sort.h>

void sah_kd_tree::Builder::populateLeafNodeTriangleRange()
{
    thrust::sort_by_key(polygon.node.begin(), polygon.node.end(), polygon.triangle.begin());

    leaf.node.resize(leaf.count);
    thrust::device_vector<U> leafPolygonCount(leaf.count);
    auto leafPolygonCountEnd = thrust::reduce_by_key(polygon.node.begin(), polygon.node.end(), thrust::make_constant_iterator<U>(1), leaf.node.begin(), leafPolygonCount.begin());
    // erase window for empty leaf nodes:
    leaf.node.erase(leafPolygonCountEnd.first, leaf.node.end());
    leafPolygonCount.erase(leafPolygonCountEnd.second, leafPolygonCount.end());

    thrust::device_vector<U> leafPolygonOffset(leafPolygonCount.size());
    thrust::exclusive_scan(leafPolygonCount.cbegin(), leafPolygonCount.cend(), leafPolygonOffset.begin());

    auto leafPolygonBegin = thrust::make_zip_iterator(leafPolygonOffset.cbegin(), leafPolygonCount.cbegin());
    auto leafPolygonEnd = thrust::make_zip_iterator(leafPolygonOffset.cend(), leafPolygonCount.cend());
    auto leafPolygonOutputBegin = thrust::make_zip_iterator(node.leftChild.begin(), node.rightChild.begin());
    thrust::scatter(leafPolygonBegin, leafPolygonEnd, leaf.node.cbegin(), leafPolygonOutputBegin);
}

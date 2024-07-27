#include <sah_kd_tree/sah_kd_tree.cuh>

#include <thrust/device_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/sort.h>

template<typename Traits>
SAH_KD_TREE_INLINE void sah_kd_tree::Builder<Traits>::populateLeafNodeTriangleRange()
{
    thrust::sort_by_key(polygon.node.begin(), polygon.node.end(), polygon.triangle.begin());

    leaf.node.resize(leaf.count);
    leaf.polygonCount.resize(leaf.count);
    auto leafPolygonCountEnd = thrust::reduce_by_key(polygon.node.begin(), polygon.node.end(), thrust::make_constant_iterator<U>(1), leaf.node.begin(), leaf.polygonCount.begin());
    // erase window for empty leaf nodes:
    leaf.node.erase(leafPolygonCountEnd.first, leaf.node.end());
    leaf.polygonCount.erase(leafPolygonCountEnd.second, leaf.polygonCount.end());

    leaf.polygonOffset.resize(leaf.polygonCount.size());
    thrust::exclusive_scan(leaf.polygonCount.cbegin(), leaf.polygonCount.cend(), leaf.polygonOffset.begin());

    auto leafPolygonBegin = thrust::make_zip_iterator(leaf.polygonOffset.cbegin(), leaf.polygonCount.cbegin());
    auto leafPolygonEnd = thrust::make_zip_iterator(leaf.polygonOffset.cend(), leaf.polygonCount.cend());
    auto leafPolygonOutputBegin = thrust::make_zip_iterator(node.leftChild.begin(), node.rightChild.begin());
    thrust::scatter(leafPolygonBegin, leafPolygonEnd, leaf.node.cbegin(), leafPolygonOutputBegin);
}

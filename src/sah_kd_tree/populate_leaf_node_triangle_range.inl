#include <common/config.hpp>
#include <sah_kd_tree/sah_kd_tree.cuh>

#include <thrust/device_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

template<typename Traits>
void sah_kd_tree::Builder<Traits>::populateLeafNodeTriangleRange()
{
    if (sah_kd_tree::kIsDebugBuild) {
        auto polygonBegin = thrust::make_zip_iterator(polygon.node.begin(), polygon.triangle.begin());
        auto polygonEnd = thrust::make_zip_iterator(polygon.node.end(), polygon.triangle.end());
        thrust::sort(exec, polygonBegin, polygonEnd);
        // uniqueness should be guaranteed in natural way by correct intersection of convex polyhedra (boxes) and convex polygones (tris)
        assert(thrust::unique(exec, polygonBegin, polygonEnd) == polygonEnd);
    } else {
        thrust::sort_by_key(exec, polygon.node.begin(), polygon.node.end(), polygon.triangle.begin());
    }

    leaf.node.resize(leaf.count);
    leaf.polygonCount.resize(leaf.count);
    auto leafPolygonCountEnd = thrust::reduce_by_key(exec, polygon.node.begin(), polygon.node.end(), thrust::make_constant_iterator<U>(1), leaf.node.begin(), leaf.polygonCount.begin());
    // erase window for empty leaf nodes:
    leaf.node.erase(leafPolygonCountEnd.first, leaf.node.end());
    leaf.polygonCount.erase(leafPolygonCountEnd.second, leaf.polygonCount.end());

    leaf.polygonOffset.resize(leaf.polygonCount.size());
    thrust::exclusive_scan(exec, leaf.polygonCount.cbegin(), leaf.polygonCount.cend(), leaf.polygonOffset.begin());

    auto leafPolygonBegin = thrust::make_zip_iterator(leaf.polygonOffset.cbegin(), leaf.polygonCount.cbegin());
    auto leafPolygonEnd = thrust::make_zip_iterator(leaf.polygonOffset.cend(), leaf.polygonCount.cend());
    auto leafPolygonOutputBegin = thrust::make_zip_iterator(node.leftChild.begin(), node.rightChild.begin());
    thrust::scatter(exec, leafPolygonBegin, leafPolygonEnd, leaf.node.cbegin(), leafPolygonOutputBegin);
}

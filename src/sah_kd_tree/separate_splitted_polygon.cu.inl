#include <sah_kd_tree/sah_kd_tree.cuh>

#include <thrust/advance.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/memory.h>

#include <cassert>

template<typename Traits>
void sah_kd_tree::Builder<Traits>::separateSplittedPolygon()
{
    polygon.triangle.resize(polygon.count + polygon.splittedCount);
    polygon.node.resize(polygon.count + polygon.splittedCount);
    splittedPolygon.resize(polygon.splittedCount);

    auto polygonBegin = thrust::make_counting_iterator<U>(0);

    auto polygonNodes = thrust::raw_pointer_cast(polygon.node.data());
    auto nodeSplitDimensions = thrust::raw_pointer_cast(node.splitDimension.data());
    auto polygonSides = thrust::raw_pointer_cast(polygon.side.data());
    U layerBase = layer.base;
    const auto isSplittedPolygon = [layerBase, polygonNodes, nodeSplitDimensions, polygonSides] __host__ __device__(U polygonIn) -> bool
    {
        U polygonNode = polygonNodes[polygonIn];
        if (polygonNode < layerBase) {
            return false;
        }
        if (nodeSplitDimensions[polygonNode] < 0) {
            return false;
        }
        return polygonSides[polygonIn] == 0;
    };
    auto polygonTriangles = thrust::raw_pointer_cast(polygon.triangle.data());
    auto polygonTriangleAndNodeBegin = thrust::make_zip_iterator(polygon.triangle.begin(), polygon.node.begin());
    auto splittedPolygonOutputBegin = thrust::make_zip_iterator(splittedPolygon.begin(), cuda::std::next(polygonTriangleAndNodeBegin, polygon.count));
    using SplittedPolygonType = cuda::std::iter_value_t<decltype(splittedPolygonOutputBegin)>;
    const auto toSplittedPolygon = [polygonTriangles, polygonNodes] __host__ __device__(U polygonIn) -> SplittedPolygonType
    {
        return {polygonIn, {polygonTriangles[polygonIn], polygonNodes[polygonIn]}};
    };
    auto splittedPolygonInputBegin = thrust::make_transform_iterator(polygonBegin, toSplittedPolygon);
    auto splittedPolygonInputEnd = cuda::std::next(splittedPolygonInputBegin, polygon.count);
    [[maybe_unused]] auto splittedPolygonOutputEnd = thrust::copy_if(exec, splittedPolygonInputBegin, splittedPolygonInputEnd, polygonBegin, splittedPolygonOutputBegin, isSplittedPolygon);
    assert(cuda::std::next(splittedPolygonOutputBegin, polygon.splittedCount) == splittedPolygonOutputEnd);
}

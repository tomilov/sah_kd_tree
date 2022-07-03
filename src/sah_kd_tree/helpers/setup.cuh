#pragma once

#include <sah_kd_tree/sah_kd_tree_export.h>
#include <sah_kd_tree/types.cuh>
#include <sah_kd_tree/utility.cuh>

#include <thrust/advance.h>
#include <thrust/device_vector.h>
#include <thrust/distance.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

namespace sah_kd_tree::helpers
{
struct SAH_KD_TREE_EXPORT Triangles
{
    U triangleCount = 0;
    struct Component
    {
        thrust::device_vector<F> a, b, c;
    } x, y, z;
};

// For non-CUDA THRUST_DEVICE_SYSTEM using the function works fine in pure .cpp,
// but to conduct with .cpp code in case of CUDA "glue" .hpp+.cu pair is required
// (ideally .hpp should contain only C++)
template<typename TriangleIterator>
void setTriangles(Triangles & triangles, TriangleIterator triangleBegin, TriangleIterator triangleEnd)
{
    Timer timer;

    using TriangleType = IteratorValueType<TriangleIterator>;
    thrust::device_vector<TriangleType> t{triangleBegin, triangleEnd};
    triangles.triangleCount = U(t.size());
    auto transposeProjection = [triangleCount = triangles.triangleCount](typename Triangles::Component & component) {
        component.a.resize(triangleCount);
        component.b.resize(triangleCount);
        component.c.resize(triangleCount);
        return thrust::make_zip_iterator(component.a.begin(), component.b.begin(), component.c.begin());
    };
    auto transposedTriangleBegin = thrust::make_zip_iterator(transposeProjection(triangles.x), transposeProjection(triangles.y), transposeProjection(triangles.z));
    using TransposedTriangleType = IteratorValueType<decltype(transposedTriangleBegin)>;
    auto transposeTriangle = [] __host__ __device__(const TriangleType & t) -> TransposedTriangleType { return {{t.a.x, t.b.x, t.c.x}, {t.a.y, t.b.y, t.c.y}, {t.a.z, t.b.z, t.c.z}}; };
    thrust::transform(t.cbegin(), t.cend(), transposedTriangleBegin, transposeTriangle);
    timer("setTriangles");  // 5.453ms
}
}  // namespace sah_kd_tree::helpers

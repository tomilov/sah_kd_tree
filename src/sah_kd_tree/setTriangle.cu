#include "utility.cuh"

#include <SahKdTree.hpp>

#include <thrust/distance.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

void SahKdTree::Builder::setTriangle(thrust::device_ptr<const Triangle> triangleBegin, thrust::device_ptr<const Triangle> triangleEnd)
{
    Timer timer;
    auto triangleCount = thrust::distance(triangleBegin, triangleEnd);
    auto transposeProjectionTriangle = [triangleCount](auto & projection) {
        auto & triangle = projection.triangle;
        triangle.a.resize(size_t(triangleCount));
        triangle.b.resize(size_t(triangleCount));
        triangle.c.resize(size_t(triangleCount));
        return thrust::make_zip_iterator(thrust::make_tuple(triangle.a.begin(), triangle.b.begin(), triangle.c.begin()));
    };
    auto transposedTriangleBegin = thrust::make_zip_iterator(thrust::make_tuple(transposeProjectionTriangle(x), transposeProjectionTriangle(y), transposeProjectionTriangle(z)));
    using TransposedTriangleType = IteratorValueType<decltype(transposedTriangleBegin)>;
    auto transposeTriangle = [] __host__ __device__(const Triangle & t) -> TransposedTriangleType { return {{t.a.x, t.b.x, t.c.x}, {t.a.y, t.b.y, t.c.y}, {t.a.z, t.b.z, t.c.z}}; };
    thrust::transform(triangleBegin, triangleEnd, transposedTriangleBegin, transposeTriangle);
    timer("setTriangle");  // 5.453ms
}

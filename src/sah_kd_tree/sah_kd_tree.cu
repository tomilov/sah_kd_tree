#include "sah_kd_tree.cuh"

#include <sah_kd_tree/sah_kd_tree.hpp>

#include <thrust/distance.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <cassert>

#ifndef __CUDACC_EXTENDED_LAMBDA__
#error "nvcc --extended-lambda"
#endif
#ifndef __CUDACC_RELAXED_CONSTEXPR__
#error "nvcc --expt-relaxed-constexpr"
#endif

namespace SahKdTree
{
void Builder::operator()(const Params & /*sah*/)
{
    auto triangleCount = U(x.triangle.a.size());
    assert(triangleCount == U(y.triangle.a.size()));
    assert(triangleCount == U(z.triangle.a.size()));

    Timer timer;

    x.calculateTriangleBbox();
    y.calculateTriangleBbox();
    z.calculateTriangleBbox();
    timer("calculateTriangleBbox");  // 0.004484

    x.caluculateRootNodeBbox();
    y.caluculateRootNodeBbox();
    z.caluculateRootNodeBbox();
    timer("caluculateRootNodeBbox");  // 0.001709

    x.generateInitialEvent();
    y.generateInitialEvent();
    z.generateInitialEvent();
    timer("generateInitialEvent");  // 0.138127
}

void build(const Params & sah, thrust::device_ptr<const Triangle> triangleBegin, thrust::device_ptr<const Triangle> triangleEnd)
{
    Builder builder;
    Timer timer;
    auto triangleCount = thrust::distance(triangleBegin, triangleEnd);
    auto resizeTriangleProjection = [triangleCount](auto & projection) {
        auto & triangle = projection.triangle;
        triangle.a.resize(size_t(triangleCount));
        triangle.b.resize(size_t(triangleCount));
        triangle.c.resize(size_t(triangleCount));
    };
    resizeTriangleProjection(builder.x);
    resizeTriangleProjection(builder.y);
    resizeTriangleProjection(builder.z);

    auto splitTriangleProjection = [](auto & projection) {
        auto & triangle = projection.triangle;
        return thrust::make_zip_iterator(thrust::make_tuple(triangle.a.begin(), triangle.b.begin(), triangle.c.begin()));
    };
    auto splittedTriangleBegin = thrust::make_zip_iterator(thrust::make_tuple(splitTriangleProjection(builder.x), splitTriangleProjection(builder.y), splitTriangleProjection(builder.z)));
    using SplittedTriangleType = IteratorValueType<decltype(splittedTriangleBegin)>;
    auto splitTriangle = [] __host__ __device__(const Triangle & t) -> SplittedTriangleType { return {{t.a.x, t.a.y, t.a.z}, {t.b.x, t.b.y, t.b.z}, {t.c.x, t.c.y, t.c.z}}; };
    thrust::transform(triangleBegin, triangleEnd, splittedTriangleBegin, splitTriangle);
    timer("AoS to SoA");  // 0.005521
    return builder(sah);
}
}  // namespace SahKdTree

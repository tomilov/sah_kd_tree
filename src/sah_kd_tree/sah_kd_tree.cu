#include "sah_kd_tree.cuh"

#include <sah_kd_tree/sah_kd_tree.hpp>

#include <thrust/advance.h>
#include <thrust/distance.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/system/cuda/vector.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#ifndef __CUDACC_EXTENDED_LAMBDA__
#error "nvcc --extended-lambda"
#endif
#ifndef __CUDACC_RELAXED_CONSTEXPR__
#error "nvcc --expt-relaxed-constexpr"
#endif

namespace SahKdTree
{
template<typename Type>
struct AddLeafConstReference
{
    using type = const Type &;  // I, U, F
};

template<typename Type>
using AddLeafConstReferenceType = typename AddLeafConstReference<Type>::type;

template<>
struct AddLeafConstReference<thrust::null_type>
{
    using type = thrust::null_type;
};

template<typename... Types>
struct AddLeafConstReference<thrust::tuple<Types...>>
{
    using type = thrust::tuple<AddLeafConstReferenceType<Types>...>;
};

template<typename Iterator>
using InputIteratorValueType = AddLeafConstReferenceType<typename Iterator::value_type>;

template<typename Iterator>
using OutputIteratorValueType = typename Iterator::value_type;

template struct Projection<0>;
template struct Projection<1>;
template struct Projection<2>;

template<Vertex Triangle::*V, float Vertex::*C>
struct TriangleSlice
{
    __host__ __device__ float operator()(const Triangle & t)
    {
        return t.*V.*C;
    }
};

void build(const Params & sah, thrust::cpp::pointer<const Triangle> trianglesBegin, thrust::cpp::pointer<const Triangle> trianglesEnd)
{
    Builder builder;
    {
        thrust::cuda::vector<Triangle> triangles{trianglesBegin, trianglesEnd};
        auto triangleBegin{triangles.cbegin()}, triangleEnd{triangles.cend()};
        auto triangleCount = thrust::distance(triangleBegin, triangleEnd);

        if ((false)) {
            auto ax = thrust::make_transform_iterator(triangleBegin, TriangleSlice<&Triangle::a, &Vertex::x>{});
            auto ay = thrust::make_transform_iterator(triangleBegin, TriangleSlice<&Triangle::a, &Vertex::y>{});
            auto az = thrust::make_transform_iterator(triangleBegin, TriangleSlice<&Triangle::a, &Vertex::z>{});

            auto bx = thrust::make_transform_iterator(triangleBegin, TriangleSlice<&Triangle::b, &Vertex::x>{});
            auto by = thrust::make_transform_iterator(triangleBegin, TriangleSlice<&Triangle::b, &Vertex::y>{});
            auto bz = thrust::make_transform_iterator(triangleBegin, TriangleSlice<&Triangle::b, &Vertex::z>{});

            auto cx = thrust::make_transform_iterator(triangleBegin, TriangleSlice<&Triangle::c, &Vertex::x>{});
            auto cy = thrust::make_transform_iterator(triangleBegin, TriangleSlice<&Triangle::c, &Vertex::y>{});
            auto cz = thrust::make_transform_iterator(triangleBegin, TriangleSlice<&Triangle::c, &Vertex::z>{});

            auto setProjectionTriangles = [triangleCount](auto & projection, auto a, auto b, auto c) {
                auto & triangle = projection.triangle;
                triangle.a.assign(a, thrust::next(a, triangleCount));
                triangle.b.assign(b, thrust::next(b, triangleCount));
                triangle.c.assign(c, thrust::next(c, triangleCount));
            };
            setProjectionTriangles(builder.x, ax, bx, cx);
            setProjectionTriangles(builder.y, ay, by, cy);
            setProjectionTriangles(builder.z, az, bz, cz);
        } else {
            Builder builder;
            auto resizeProjectionTriangles = [triangleCount](auto & projection) {
                auto & triangle = projection.triangle;
                triangle.a.resize(size_t(triangleCount));
                triangle.b.resize(size_t(triangleCount));
                triangle.c.resize(size_t(triangleCount));
            };
            resizeProjectionTriangles(builder.x);
            resizeProjectionTriangles(builder.y);
            resizeProjectionTriangles(builder.z);

            auto getProjectionTriangles = [](auto & projection) {
                auto & triangle = projection.triangle;
                return thrust::make_zip_iterator(thrust::make_tuple(triangle.a.begin(), triangle.b.begin(), triangle.c.begin()));
            };
            auto projectedTrianglesbegin = thrust::make_zip_iterator(thrust::make_tuple(getProjectionTriangles(builder.x), getProjectionTriangles(builder.y), getProjectionTriangles(builder.z)));
            using ProjectedTrianglesType = OutputIteratorValueType<decltype(projectedTrianglesbegin)>;
            auto projectPointsCoordinates = [] __host__ __device__(const Triangle & t) -> ProjectedTrianglesType { return {{t.a.x, t.a.y, t.a.z}, {t.b.x, t.b.y, t.b.z}, {t.c.x, t.c.y, t.c.z}}; };
            thrust::transform(triangleBegin, triangleEnd, projectedTrianglesbegin, projectPointsCoordinates);
        }
    }
    return builder(sah);
}
}  // namespace SahKdTree

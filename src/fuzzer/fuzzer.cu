#include <fuzzer/fuzzer.hpp>
#include <sah_kd_tree/sah_kd_tree.cuh>
#include <utils/auto_cast.hpp>

#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <functional>
#include <iterator>
#include <type_traits>

#include <cstddef>

namespace fuzzer
{
namespace
{

template<typename Traits>
struct Vertices
{
    typename Traits::template Vector<typename Traits::F> x, y, z;

    explicit Vertices(size_t triangleCount)
        : x(3 * triangleCount)
        , y(3 * triangleCount)
        , z(3 * triangleCount)
    {}
};

template<typename Traits>
struct Indices
{
    typename Traits::template Vector<typename Traits::U> a, b, c;

    explicit Indices(size_t triangleCount)
        : a(triangleCount)
        , b(triangleCount)
        , c(triangleCount)
    {}
};

}  // namespace

void testOneInput(
    const Params & p,
    const std::vector<Triangle> & triangles)
{
    using Traits = sah_kd_tree::DefaultTraits;

    const size_t triangleCount = std::size(triangles);
    Vertices<Traits> vertices{triangleCount};
    {
        static_assert(std::is_same_v<F, typename Traits::F>);
        auto a = thrust::make_zip_iterator(vertices.x.begin(), vertices.y.begin(), vertices.z.begin());
        auto b = cuda::std::next(a);
        auto c = cuda::std::next(b);
        auto abc = thrust::make_zip_iterator(a, b, c);
        using OutputTriangle = cuda::std::iter_value_t<decltype(abc)>;
        const auto transposeTriangle = [] __host__ __device__(const Triangle & triangle) -> OutputTriangle
        {
            return {{triangle.a.x, triangle.a.y, triangle.a.z}, {triangle.b.x, triangle.b.y, triangle.b.z}, {triangle.c.x, triangle.c.y, triangle.c.z}};
        };
        thrust::transform_n(std::cbegin(triangles), utils::autoCast(triangleCount), abc, transposeTriangle);
    }
    Indices<Traits> indices{triangleCount};
    {
        static_assert(std::is_same_v<U, typename Traits::U>);
        thrust::tabulate(indices.a.begin(), indices.a.end(), [] __host__ __device__(ptrdiff_t i) { return 0 + i * 3; });
        thrust::tabulate(indices.b.begin(), indices.b.end(), [] __host__ __device__(ptrdiff_t i) { return 1 + i * 3; });
        thrust::tabulate(indices.c.begin(), indices.c.end(), [] __host__ __device__(ptrdiff_t i) { return 2 + i * 3; });
    }

    sah_kd_tree::Builder<Traits> builder;
    sah_kd_tree::Projection<Traits> x, y, z;
    {
        builder.polygon.count = utils::autoCast(triangleCount);

        x.triangle.count = builder.polygon.count;
        x.triangle.a = thrust::make_permutation_iterator(vertices.x.data(), indices.a.data());
        x.triangle.b = thrust::make_permutation_iterator(vertices.x.data(), indices.b.data());
        x.triangle.c = thrust::make_permutation_iterator(vertices.x.data(), indices.c.data());

        y.triangle.count = builder.polygon.count;
        y.triangle.a = thrust::make_permutation_iterator(vertices.y.data(), indices.a.data());
        y.triangle.b = thrust::make_permutation_iterator(vertices.y.data(), indices.b.data());
        y.triangle.c = thrust::make_permutation_iterator(vertices.y.data(), indices.c.data());

        z.triangle.count = builder.polygon.count;
        z.triangle.a = thrust::make_permutation_iterator(vertices.z.data(), indices.a.data());
        z.triangle.b = thrust::make_permutation_iterator(vertices.z.data(), indices.b.data());
        z.triangle.c = thrust::make_permutation_iterator(vertices.z.data(), indices.c.data());
    }

    const sah_kd_tree::Params<Traits> params = {
        .emptinessFactor = p.emptinessFactor,
        .traversalCost = p.traversalCost,
        .intersectionCost = p.intersectionCost,
        .maxTreeDepth = p.maxTreeDepth,
    };

    const std::function<bool(size_t progressValue)> cancel = []([[maybe_unused]] size_t progress)
    {
        return false;
    };
    sah_kd_tree::Tree<Traits> tree;
    builder.build(cancel, params, x, y, z, tree);
}
}  // namespace fuzzer

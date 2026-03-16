#include <fuzzer/fuzzer.hpp>
#include <sah_kd_tree/sah_kd_tree.cuh>
#include <utils/auto_cast.hpp>

#include <thrust/iterator/permutation_iterator.h>
#include <thrust/tabulate.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/transform.h>

#include <functional>
#include <iterator>

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

}

void testOneInput(
    const Params & p,
    const std::vector<Triangle> & t)
{
    using Traits = sah_kd_tree::DefaultTraits;

    const thrust::host_vector<Triangle> triangles{std::cbegin(t), std::cend(t)};

    const size_t triangleCount = std::size(t);
    Vertices<Traits> vertices{triangleCount};
    {
        static_assert(std::is_same_v<F, typename Traits::F>);
        auto ax = vertices.x.begin();
        auto bx = cuda::std::next(ax);
        auto cx = cuda::std::next(bx);
        auto abcx = thrust::make_zip_iterator(ax, bx, cx);
        auto ay = vertices.y.begin();
        auto by = cuda::std::next(ay);
        auto cy = cuda::std::next(by);
        auto abcy = thrust::make_zip_iterator(ay, by, cy);
        auto az = vertices.z.begin();
        auto bz = cuda::std::next(az);
        auto cz = cuda::std::next(bz);
        auto abcz = thrust::make_zip_iterator(az, bz, cz);
        auto triangle = thrust::make_zip_iterator(abcx, abcy, abcz);
        using OutputTriangle = cuda::std::iter_value_t<decltype(triangle)>;
        const auto transposeTriangle = [] __host__ __device__(const Triangle & triangle) -> OutputTriangle
        {
            return {{triangle.a.x, triangle.b.x, triangle.c.x}, {triangle.a.y, triangle.b.y, triangle.c.y}, {triangle.a.z, triangle.b.z, triangle.c.z}};
        };
        thrust::transform_n(triangles.cbegin(), utils::autoCast(triangleCount), triangle, transposeTriangle);
    }
    Indices<Traits> indices{triangleCount};
    {
        static_assert(std::is_same_v<U, typename Traits::U>);
        thrust::tabulate(indices.a.begin(), indices.a.end(), []__host__ __device__(ptrdiff_t i){ return 0 + i * 3; });
        thrust::tabulate(indices.b.begin(), indices.b.end(), []__host__ __device__(ptrdiff_t i){ return 1 + i * 3; });
        thrust::tabulate(indices.c.begin(), indices.c.end(), []__host__ __device__(ptrdiff_t i){ return 2 + i * 3; });
    }

    sah_kd_tree::Builder<Traits> builder;
    sah_kd_tree::Projection<Traits> x, y, z;
    {
        builder.polygon.count = utils::autoCast(triangleCount);

        x.triangle.count = builder.polygon.count;
        x.triangle.a = thrust::make_permutation_iterator(vertices.x.cbegin(), indices.a.cbegin());
        x.triangle.b = thrust::make_permutation_iterator(vertices.x.cbegin(), indices.b.cbegin());
        x.triangle.c = thrust::make_permutation_iterator(vertices.x.cbegin(), indices.c.cbegin());

        y.triangle.count = builder.polygon.count;
        y.triangle.a = thrust::make_permutation_iterator(vertices.y.cbegin(), indices.a.cbegin());
        y.triangle.b = thrust::make_permutation_iterator(vertices.y.cbegin(), indices.b.cbegin());
        y.triangle.c = thrust::make_permutation_iterator(vertices.y.cbegin(), indices.c.cbegin());

        z.triangle.count = builder.polygon.count;
        z.triangle.a = thrust::make_permutation_iterator(vertices.z.cbegin(), indices.a.cbegin());
        z.triangle.b = thrust::make_permutation_iterator(vertices.z.cbegin(), indices.b.cbegin());
        z.triangle.c = thrust::make_permutation_iterator(vertices.z.cbegin(), indices.c.cbegin());
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

#include <SahKdTree.hpp>

#include <thrust/device_ptr.h>
#include <fuzzer/FuzzedDataProvider.h>

#include <iterator>
#include <vector>
#include <tuple>
#include <type_traits>
#include <limits>

#include <cstdint>
#include <cstdlib>
#include <cmath>

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
#error "Not supported!"
#endif

extern "C" int LLVMFuzzerTestOneInput(const uint8_t * data, size_t size)
{
    using namespace SahKdTree;
    static_assert(std::is_integral_v<I>, "!");
    static_assert(std::is_integral_v<U>, "!");
    static_assert(std::is_floating_point_v<F>, "!");

    if (size == 0) {
        return 0;
    }

    Params params;
    params.maxDepth = U(size) + 1;
    std::vector<Triangle> triangles;
    {
        FuzzedDataProvider fdp{data, size};
        params.emptinessFactor = fdp.ConsumeFloatingPointInRange<F>(F(0.125), F(1));
        params.traversalCost = fdp.ConsumeFloatingPointInRange<F>(F(1), F(1000));
        params.intersectionCost = fdp.ConsumeFloatingPointInRange<F>(F(1), F(1000));
        const auto consumeAlmostInteger = [&]() -> F
        {
            const int power = fdp.ConsumeIntegralInRange<int>(2, std::numeric_limits<F>::digits);
            const F fuzz = std::ldexp(F(1) + fdp.ConsumeProbability<F>(), -power);
            return fuzz + F(fdp.ConsumeIntegralInRange<int>(-5, 5));
        };
        const auto consumeVertex = [&](Vertex & vertex) -> std::tuple<F, F, F>
        {
            vertex.x = consumeAlmostInteger();
            vertex.y = consumeAlmostInteger();
            vertex.z = consumeAlmostInteger();
            return {vertex.x, vertex.y, vertex.z};
        };
        while (fdp.remaining_bytes() == 0) {
            Triangle & triangle = triangles.emplace_back();
            const auto a = consumeVertex(triangle.a);
            const auto b = consumeVertex(triangle.b);
            if (b == a) {
                return 0;
            }
            const auto c = consumeVertex(triangle.c);
            if (c == a) {
                return 0;
            }
            if (c == b) {
                return 0;
            }
        }
        if (triangles.empty()) {
            return 0;
        }
    }

    Builder builder;
    builder.setTriangle(thrust::device_pointer_cast(triangles.data()), thrust::device_pointer_cast(triangles.data() + triangles.size()));
    Tree tree = builder(params);
    if (tree.depth >= params.maxDepth) {
        std::abort();
    }
    return 0;
}

#include <SahKdTree.hpp>

#include <thrust/device_ptr.h>

#include <algorithm>
#include <array>
#include <bitset>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <limits>
#include <numeric>
#include <random>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <cassert>
#include <cstdlib>

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
#error "CUDA backend is not supported"
#endif

namespace SahKdTree
{
inline bool operator<(const Vertex & lhs, const Vertex & rhs)
{
    return std::tie(lhs.x, lhs.y, lhs.z) < std::tie(rhs.x, rhs.y, rhs.z);
}

inline bool operator<(const Triangle & lhs, const Triangle & rhs)
{
    return std::tie(lhs.a, lhs.b, lhs.c) < std::tie(rhs.a, rhs.b, rhs.c);
}
}  // namespace SahKdTree

using namespace SahKdTree;

static_assert(std::is_integral_v<I>, "!");
static_assert(std::is_integral_v<U>, "!");
static_assert(std::is_floating_point_v<F>, "!");

namespace
{
constexpr size_t maxTriangleCount = 1;
constexpr int intBboxSize = 5;
constexpr bool sortTriangles = true;
constexpr bool fuzzIntegerCoordinate = false;

constexpr int floatDigits = std::numeric_limits<F>::digits;

std::mt19937 gen;
std::uniform_int_distribution<> uniformInt;

using UniformIntParam = typename decltype(uniformInt)::param_type;

void SetSeed(unsigned int seed)
{
    using SeedType = typename std::mt19937::result_type;
    gen.seed(SeedType(seed));
}

void generateComponent(F & f)
{
    f = F(uniformInt(gen, UniformIntParam(-intBboxSize, +intBboxSize)));
    if constexpr (fuzzIntegerCoordinate) {
        const int pow = uniformInt(gen, UniformIntParam(1, floatDigits));
        auto fuzz = std::generate_canonical<F, floatDigits>(gen);
        fuzz += fuzz;
        fuzz -= F(1);  // lost 1 bit of mantissa's randomness
        f += std::scalbn(fuzz, -pow);
    }
}

void generateVertex(Vertex & vertex)
{
    auto & [x, y, z] = vertex;
    generateComponent(x);
    generateComponent(y);
    generateComponent(z);
}

void generateTriangle(Triangle & triangle)
{
    auto & [a, b, c] = triangle;
    generateVertex(a);
    generateVertex(b);
    generateVertex(c);

    if constexpr (sortTriangles) {
        if (b < a) {
            std::swap(a, b);
        }
        if (c < a) {
            std::swap(a, c);
        }
        if (c < b) {
            std::swap(b, c);
        }
    }
}

struct TestInput
{
    Params params;
    std::vector<Triangle> triangles;

    static_assert(std::is_standard_layout_v<Params> && std::is_trivially_copyable_v<Params>, "!");
    static_assert(std::is_standard_layout_v<Triangle> && std::is_trivially_copyable_v<Triangle>, "!");

    static constexpr size_t paramsSize = sizeof params.emptinessFactor + sizeof params.traversalCost + sizeof params.intersectionCost;

    void generateTriangles(size_t triangleCount)
    {
        triangles.resize(triangleCount);
        for (Triangle & triangle : triangles) {
            generateTriangle(triangle);
        }
        if constexpr (sortTriangles) {
            std::sort(std::begin(triangles), std::end(triangles));
        }
    }

    void generate(size_t triangleCount = 1)
    {
        params.emptinessFactor = std::generate_canonical<F, floatDigits>(gen);
        params.traversalCost = std::generate_canonical<F, floatDigits>(gen);
        params.intersectionCost = std::generate_canonical<F, floatDigits>(gen);

        return generateTriangles(triangleCount);
    }

    bool read(const uint8_t * data, size_t size)
    {
        if (size < paramsSize + sizeof(Triangle)) {
            return false;
        }

        std::memcpy(&params.emptinessFactor, data, sizeof params.emptinessFactor);
        if (std::isnan(params.emptinessFactor) || (params.emptinessFactor < F(0)) || (params.emptinessFactor > F(1))) {
            return false;
        }
        data += sizeof params.emptinessFactor;
        size -= sizeof params.emptinessFactor;

        std::memcpy(&params.traversalCost, data, sizeof params.traversalCost);
        if (std::isnan(params.traversalCost) || (params.traversalCost < F(0))) {
            return false;
        }
        data += sizeof params.traversalCost;
        size -= sizeof params.traversalCost;

        std::memcpy(&params.intersectionCost, data, sizeof params.intersectionCost);
        if (std::isnan(params.intersectionCost) || (params.intersectionCost < F(0))) {
            return false;
        }
        data += sizeof params.intersectionCost;
        size -= sizeof params.intersectionCost;

        triangles.resize(size / sizeof(Triangle));
        std::memcpy(std::data(triangles), data, size);
        for (const Triangle & triangle : triangles) {
            if (std::isnan(triangle.a.x)) return false;
            if (std::isnan(triangle.a.y)) return false;
            if (std::isnan(triangle.a.z)) return false;

            if (std::isnan(triangle.b.x)) return false;
            if (std::isnan(triangle.b.y)) return false;
            if (std::isnan(triangle.b.z)) return false;

            if (std::isnan(triangle.c.x)) return false;
            if (std::isnan(triangle.c.y)) return false;
            if (std::isnan(triangle.c.z)) return false;
        }
        if constexpr (sortTriangles) {
            if (!std::is_sorted(std::cbegin(triangles), std::cend(triangles))) {
                return false;
            }
        }
        return true;
    }

    size_t write(uint8_t * data, size_t maxSize) const  // lossy: tail can be cropped
    {
        if (maxSize < paramsSize + sizeof(Triangle)) {
            std::exit(EXIT_FAILURE);
        }

        size_t size = 0;

        std::memcpy(data, &params.emptinessFactor, sizeof params.emptinessFactor);
        data += sizeof params.emptinessFactor;
        size += sizeof params.emptinessFactor;

        std::memcpy(data, &params.traversalCost, sizeof params.traversalCost);
        data += sizeof params.traversalCost;
        size += sizeof params.traversalCost;

        std::memcpy(data, &params.intersectionCost, sizeof params.intersectionCost);
        data += sizeof params.intersectionCost;
        size += sizeof params.intersectionCost;

        std::vector<typename decltype(triangles)::const_iterator> excluded;
        if (const size_t maxCount = (maxSize - size) / sizeof(Triangle); maxCount < std::size(triangles)) {
            excluded.resize(std::size(triangles));
            std::iota(std::begin(excluded), std::end(excluded), std::cbegin(triangles));
            std::shuffle(std::begin(excluded), std::end(excluded), gen);
            excluded.resize(std::size(triangles) - maxCount);
            std::sort(std::begin(excluded), std::end(excluded));
            assert(std::adjacent_find(std::cbegin(excluded), std::cend(excluded)) == std::cend(excluded));
        }
        auto triangleBeg = std::cbegin(triangles);
        for (const auto triangleEnd : excluded) {
            if (triangleBeg < triangleEnd) {
                assert(triangleEnd != std::cend(triangles));
                const size_t chunkSize = size_t(std::distance(triangleBeg, triangleEnd)) * sizeof(Triangle);
                std::memcpy(data, &*triangleBeg, chunkSize);
                data += chunkSize;
                size += chunkSize;
            } else {
                assert(triangleBeg == triangleEnd);
            }
            triangleBeg = std::next(triangleEnd);
        }
        if (triangleBeg != std::cend(triangles)) {
            const size_t chunkSize = size_t(std::distance(triangleBeg, std::cend(triangles))) * sizeof(Triangle);
            std::memcpy(data, &*triangleBeg, chunkSize);
            data += chunkSize;
            size += chunkSize;
        }
        assert(size <= maxSize);
        return size;
    }

    void mutateParams(ptrdiff_t action)
    {
        switch (action) {
        case 0:
            params.emptinessFactor = std::generate_canonical<F, floatDigits>(gen);
            break;
        case 1:
            params.traversalCost = std::generate_canonical<F, floatDigits>(gen);
            break;
        case 2:
            params.intersectionCost = std::generate_canonical<F, floatDigits>(gen);
            break;
        default:
            break;  // no-op
        }
    }

    void mutate(ptrdiff_t action)
    {
        const auto mutateTriangle = [this](const auto t) {
            generateTriangle(*t);
            if constexpr (sortTriangles) {
                if ((t != std::begin(triangles)) && (*t < *std::prev(t))) {
                    std::rotate(std::upper_bound(std::begin(triangles), t, *t), t, std::next(t));
                } else if ((std::next(t) != std::cend(triangles)) && (*std::next(t) < *t)) {
                    std::rotate(t, std::next(t), std::upper_bound(std::next(t), std::end(triangles), *t));
                }
            }
        };
        switch (action) {
        case 0: {  // remove triangle
            if (std::size(triangles) > 1) {
                triangles.pop_back();
            } else {
                mutateTriangle(std::begin(triangles));
            }
            break;
        }
        case 1: {  // add triangle
            triangles.emplace_back();
            mutateTriangle(std::prev(std::end(triangles)));
            break;
        }
        case 2: {  // mutate triangle
            assert(!std::empty(triangles));
            mutateTriangle(std::next(std::begin(triangles), uniformInt(gen, UniformIntParam(0, int(std::size(triangles) - 1)))));
            break;
        }
        case 3:
        case 4:
        case 5: {
            return mutateParams(action - 3);
        }
        default: {
            assert(false);  // no no-op
        }
        }
    }

    void mutate()
    {
        const auto cdf = [](auto probabilities) constexpr
        {
            std::inclusive_scan(std::cbegin(probabilities), std::cend(probabilities), std::begin(probabilities));
            return probabilities;
        };
        constexpr auto action = cdf(std::to_array({1.0f, 1.0f, 1.0f, 0.1f, 0.1f, 0.1f}));
        const float probability = std::generate_canonical<float, std::numeric_limits<float>::digits>(gen);
        mutate(std::distance(std::cbegin(action), std::upper_bound(std::cbegin(action), std::cend(action), probability * action.back())));
    }

    void cross(const TestInput & testInput)
    {
        std::bitset<3> selector{gen()};
        if (selector[0]) {
            params.emptinessFactor = testInput.params.emptinessFactor;
        }
        if (selector[1]) {
            params.traversalCost = testInput.params.traversalCost;
        }
        if (selector[2]) {
            params.intersectionCost = testInput.params.intersectionCost;
        }

        decltype(triangles) allTriangles(std::size(triangles) + std::size(testInput.triangles));
        [[maybe_unused]] const auto tend = std::merge(std::cbegin(triangles), std::cend(triangles), std::cbegin(testInput.triangles), std::cend(testInput.triangles), std::begin(allTriangles));
        assert(tend == std::end(allTriangles));
        std::swap(triangles, allTriangles);
    }
};

char * args[100];
char maxLenOption[100] = "-max_len=";
[[maybe_unused]] char timeoutOption[] = "-timeout=1";
char * options[] = {
    maxLenOption,
#ifdef NDEBUG
    timeoutOption,
#endif
};
}  // namespace

extern "C"
{
    int LLVMFuzzerInitialize(int * argc, char *** argv)
    {
        if (size_t(*argc) + std::extent_v<decltype(options)> > std::extent_v<decltype(args)>) {
            std::exit(EXIT_FAILURE);
        }

        const auto maxLen = TestInput::paramsSize + maxTriangleCount * sizeof(Triangle);
        auto [p, ec] = std::to_chars(std::next(std::begin(maxLenOption), std::strlen(maxLenOption)), std::end(maxLenOption), maxLen);
        if (ec != std::errc{}) {
            std::exit(EXIT_FAILURE);
        }
        *p = '\0';

        auto arg = args;
        *arg++ = **argv;
        for (char * a : options) {
            *arg++ = a;
            ++*argc;
        }
        for (int i = 1; i < *argc; ++i) {
            *arg++ = (*argv)[i];
        }
        *argv = args;
        return 0;
    }

    size_t LLVMFuzzerCustomMutator(uint8_t * data, size_t size, size_t maxSize, unsigned int seed)
    {
        SetSeed(seed);

        TestInput testInput;
        if (testInput.read(data, size)) {
            testInput.mutate();
        } else {
            testInput.generate();
        }
        return testInput.write(data, maxSize);
    }

    size_t LLVMFuzzerCustomCrossOver(const uint8_t * data1, size_t size1, const uint8_t * data2, size_t size2, uint8_t * out, size_t maxOutSize, unsigned int seed)
    {
        SetSeed(seed);

        TestInput testInput1;
        if (!testInput1.read(data1, size1)) {
            testInput1.generate();
        }
        TestInput testInput2;
        if (!testInput2.read(data2, size2)) {
            testInput2.generate();
        }
        testInput1.cross(testInput2);
        return testInput1.write(out, maxOutSize);
    }

    int LLVMFuzzerTestOneInput(const uint8_t * data, size_t size)
    {
        TestInput testInput;
        if (!testInput.read(data, size)) {
            return 0;
        }

        const auto & params = testInput.params;
        const auto & triangles = testInput.triangles;

        Builder builder;
        builder.setTriangle(thrust::device_pointer_cast(std::data(triangles)), thrust::device_pointer_cast(std::data(triangles) + std::size(triangles)));
        Tree tree = builder(params);
        if (!(tree.depth < std::size(triangles) + 100)) {
            std::abort();
        }
        return 0;
    }
}

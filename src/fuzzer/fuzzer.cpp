#include <fuzzer/fuzzer.hpp>

#include <fmt/color.h>
#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <bitset>
#include <charconv>
#include <functional>
#include <iterator>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>

using namespace std::rel_ops;

namespace fuzzer
{
namespace
{
constexpr size_t kBoxTriangleCount = 12;

constexpr int kIntBboxSize = 10;
constexpr bool kFuzzIntegerCoordinate = false;

constexpr int kFloatDigits = std::numeric_limits<F>::digits;

using RandomValueType = typename std::mt19937::result_type;
using UniformIntDistribution = std::uniform_int_distribution<>;
using UniformIntParam = typename UniformIntDistribution::param_type;

bool boxWorld = false;

std::mt19937 gen;                   // clazy:exclude=non-pod-global-static
UniformIntDistribution uniformInt;  // clazy:exclude=non-pod-global-static

void setSeed(unsigned int seed)
{
    gen.seed(RandomValueType(seed));
}

F genFloat()
{
    return std::generate_canonical<F, kFloatDigits>(gen);
}

void genComponent(F & f, int min = 0, int max = +kIntBboxSize)
{
    assert(!(max < min));
    f = F(uniformInt(gen, UniformIntParam(min, max)));
    if (kFuzzIntegerCoordinate) {
        const int pow = uniformInt(gen, UniformIntParam(1, kFloatDigits));
        auto fuzz = genFloat();
        fuzz += fuzz;
        fuzz -= F(1);  // lost 1 bit of mantissa's randomness
        f += std::scalbn(fuzz, -pow);
        if (f < F(min)) {
            f += F(max - min);
        } else if (F(max) < f) {
            f += F(min - max);
        }
    }
}

void genVertex(Vertex & vertex)
{
    auto & [x, y, z] = vertex;
    genComponent(x);
    genComponent(y);
    genComponent(z);
}

void genTriangle(Triangle & triangle)
{
    auto & [a, b, c] = triangle;
    genVertex(a);
    genVertex(b);
    genVertex(c);

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

template<typename TriangleOutputIterator>
TriangleOutputIterator putBox(TriangleOutputIterator out, const Vertex & a, const Vertex & b)
{
    assert(a.x < b.x);
    assert(a.y < b.y);
    assert(a.z < b.z);
    Vertex v[] = {a, a, a, b, b, b};
    v[0].z = b.z;
    v[1].y = b.y;
    v[2].x = b.x;
    v[3].x = a.x;
    v[4].y = a.y;
    v[5].z = a.z;
    *out++ = {a, v[0], v[3]};
    *out++ = {a, v[0], v[4]};
    *out++ = {a, v[1], v[3]};
    *out++ = {a, v[1], v[5]};
    *out++ = {a, v[2], v[4]};
    *out++ = {a, v[2], v[5]};
    *out++ = {v[0], v[3], b};
    *out++ = {v[0], v[4], b};
    *out++ = {v[1], v[3], b};
    *out++ = {v[1], v[5], b};
    *out++ = {v[2], v[4], b};
    *out++ = {v[2], v[5], b};
    return out;
}

template<typename TriangleOutputIterator>
TriangleOutputIterator generateBox(TriangleOutputIterator out)
{
    Vertex min, max;
    do {
        genVertex(min);
        genVertex(max);
        std::tie(min.x, max.x) = std::minmax({min.x, max.x});
        std::tie(min.y, max.y) = std::minmax({min.y, max.y});
        std::tie(min.z, max.z) = std::minmax({min.z, max.z});
    } while (!(min.x < max.x) || !(min.y < max.y) || !(min.z < max.z));
    putBox(out, min, max);
    return out;
}

template<typename TriangleRandomAccessIterator>
bool checkItems(TriangleRandomAccessIterator beg, TriangleRandomAccessIterator end)
{
    if (boxWorld) {
        assert((std::distance(beg, end) % kBoxTriangleCount) == 0);
        for (auto box = beg; box != end; std::advance(box, kBoxTriangleCount)) {
            if (!std::is_sorted(box, std::next(box, kBoxTriangleCount))) {
                return false;
            }
            auto min = box[0].a;
            auto max = std::next(box, kBoxTriangleCount / 2)->c;
            if (!(min.x < max.x) || !(min.y < max.y) || !(min.z < max.z)) {  // consider only non-degenerate boxes
                return false;
            }
            Triangle recoveredBox[kBoxTriangleCount];
            putBox(std::begin(recoveredBox), min, max);
            if (!std::equal(std::cbegin(recoveredBox), std::cend(recoveredBox), box)) {
                return false;
            }
        }
    }
    return true;
}

size_t trianglesPerItem()
{
    return boxWorld ? kBoxTriangleCount : 1;
}

size_t itemSize()
{
    return sizeof(Triangle) * trianglesPerItem();
}

const char * primitiveName()
{
    return boxWorld ? "boxes" : "triangles";
}

struct TestInput
{
    Params params;
    std::vector<Triangle> triangles;

    static_assert(std::is_standard_layout_v<Params> && std::is_trivially_copyable_v<Params>, "!");
    static_assert(std::is_standard_layout_v<Triangle> && std::is_trivially_copyable_v<Triangle>, "!");

    void generate(size_t triangleCount = trianglesPerItem())
    {
        assert(0 < triangleCount);
        assert((triangleCount % trianglesPerItem()) == 0);
        params.emptinessFactor = genFloat();
        params.traversalCost = genFloat();
        params.intersectionCost = genFloat();
        params.maxDepth = std::numeric_limits<U>::max();

        triangles.reserve(triangleCount);
        triangles.clear();
        while (std::size(triangles) < triangleCount) {
            if (std::size(triangles) + kBoxTriangleCount <= triangleCount) {  // add AA paralellotope
                generateBox(std::back_inserter(triangles));
                assert(std::is_sorted(std::prev(std::cend(triangles), kBoxTriangleCount), std::cend(triangles)));
            } else if (std::size(triangles) + 4 <= triangleCount) {  // add tetrahedron
                assert(!boxWorld);
                Vertex v[4];
                for (Vertex & vertex : v) {
                    genVertex(vertex);
                }
                triangles.push_back({v[1], v[2], v[3]});
                triangles.push_back({v[2], v[3], v[0]});
                triangles.push_back({v[3], v[0], v[1]});
                triangles.push_back({v[0], v[1], v[2]});
            } else {
                assert(!boxWorld);
                genTriangle(triangles.emplace_back());
            }
        }
    }

    bool read(const uint8_t * data, size_t size)
    {
        if (size < sizeof(Params) + itemSize()) {
            return false;
        }
        if (((size - sizeof(Params)) % itemSize()) != 0) {
            return false;
        }

        std::memcpy(&params, data, sizeof params);
        data += sizeof params;
        size -= sizeof params;

        if (std::isnan(params.emptinessFactor) || (params.emptinessFactor < F(0)) || (params.emptinessFactor > F(1))) {
            return false;
        }
        if (std::isnan(params.traversalCost) || (params.traversalCost < F(0))) {
            return false;
        }
        if (std::isnan(params.intersectionCost) || (params.intersectionCost < F(0))) {
            return false;
        }
        assert(params.maxDepth == std::numeric_limits<U>::max());
        if (params.maxDepth == 0) {
            return false;
        }

        triangles.resize(size / sizeof(Triangle));
        std::memcpy(std::data(triangles), data, size);
        for (const Triangle & triangle : triangles) {
            if (std::isnan(triangle.a.x)) {
                return false;
            }
            if (std::isnan(triangle.a.y)) {
                return false;
            }
            if (std::isnan(triangle.a.z)) {
                return false;
            }

            if (std::isnan(triangle.b.x)) {
                return false;
            }
            if (std::isnan(triangle.b.y)) {
                return false;
            }
            if (std::isnan(triangle.b.z)) {
                return false;
            }

            if (std::isnan(triangle.c.x)) {
                return false;
            }
            if (std::isnan(triangle.c.y)) {
                return false;
            }
            if (std::isnan(triangle.c.z)) {
                return false;
            }
        }
        if (!checkItems(std::cbegin(triangles), std::cend(triangles))) {
            return false;
        }
        return true;
    }

    size_t write(uint8_t * data, size_t maxSize) const  // Possibly lossy if triangles not fit in maxSize.
    {
        assert(!(maxSize < sizeof(Params) + itemSize()));
        assert(checkItems(std::cbegin(triangles), std::cend(triangles)));

        size_t size = 0;

        std::memcpy(data, &params, sizeof params);
        data += sizeof params;
        size += sizeof params;

        if (const size_t maxItemCount = (maxSize - size) / itemSize(); maxItemCount < std::size(triangles) / trianglesPerItem()) {
            std::vector<typename decltype(triangles)::const_iterator> servived;
            servived.reserve(std::size(triangles) / trianglesPerItem());
            for (auto t = std::begin(triangles); t != std::end(triangles); std::advance(t, trianglesPerItem())) {
                servived.push_back(t);
            }
            std::shuffle(std::begin(servived), std::end(servived), gen);
            servived.resize(maxItemCount);
            for (auto t : servived) {
                std::memcpy(data, &*t, itemSize());
                data += itemSize();
                size += itemSize();
            }
            assert(((size - sizeof params) % itemSize()) == 0);
        } else {
            const auto chunkSize = std::size(triangles) * sizeof(Triangle);
            std::memcpy(data, std::data(triangles), chunkSize);
            data += chunkSize;  // NOLINT(clang-analyzer-deadcode.DeadStores)
            size += chunkSize;
        }
        assert(size <= maxSize);
        return size;
    }

    void mutateTriangles(ptrdiff_t action)
    {
        assert(!boxWorld);
        const auto getTriangle = [this] {
            assert(!std::empty(triangles));
            return std::next(std::begin(triangles), uniformInt(gen, UniformIntParam(0, int(std::size(triangles) - 1))));
        };
        const auto mutateTriangle = [&](auto t) { genTriangle(*t); };
        switch (action) {
        case 0: {
            params.emptinessFactor = genFloat();
            break;
        }
        case 1: {
            params.traversalCost = genFloat();
            break;
        }
        case 2: {
            params.intersectionCost = genFloat();
            break;
        }
        case 3: {  // Remove triangle.
            if (std::size(triangles) > 1) {
                const auto t = getTriangle();
                std::iter_swap(t, std::prev(std::end(triangles)));
                triangles.pop_back();
            } else {
                mutateTriangle(std::begin(triangles));
            }
            break;
        }
        case 4: {  // Add triangle.
            triangles.emplace_back();
            mutateTriangle(std::prev(std::end(triangles)));
            break;
        }
        case 5: {  // Mutate triangle.
            mutateTriangle(getTriangle());
            break;
        }
        case 6: {  // Make one or 3 sides of triangle axis perpendicular to an ort.
            const auto t = getTriangle();

            auto selector = gen();

            const bool singleComponent = (selector & 1) == 0;
            selector >>= 1;

            const bool fullTriangle = (selector & 1) == 0;
            selector >>= 1;

            Vertex * vertices[] = {&t->a, &t->b, &t->c};
            std::rotate(std::begin(vertices), std::next(std::begin(vertices), ptrdiff_t(selector % 3)), std::end(vertices));
            selector /= 3;

            F Vertex::*components[] = {&Vertex::x, &Vertex::y, &Vertex::z};
            std::rotate(std::begin(components), std::next(std::begin(components), ptrdiff_t(selector % 3)), std::end(components));
            // selector /= 3;

            const auto [a, b, c] = vertices;
            const auto [x, y, z] = components;
            if (fullTriangle) {
                a->*x = b->*x = c->*x;
            } else {
                a->*x = b->*x;
                if (!singleComponent) {
                    a->*y = b->*y;
                }
            }
            break;
        }
        case 7: {  // Make vertex or leg common for two triangles.
            if (std::size(triangles) == 1) {
                break;
            }

            auto src = getTriangle();
            auto dst = getTriangle();

            auto selector = gen();
            if (src == dst) {
                if ((selector & 1) == 0) {
                    if (++dst == std::end(triangles)) {
                        dst = std::prev(src);
                    }
                } else {
                    if (dst != std::begin(triangles)) {
                        --dst;
                    } else {
                        ++dst;
                    }
                }
                selector >>= 1;
            }

            std::bitset<2> direction(selector);
            selector >>= 2;

            const auto s = selector % 3;
            selector /= 3;

            const auto d = selector % 3;
            selector /= 3;

            constexpr Vertex Triangle::*vertices[] = {&Triangle::a, &Triangle::b, &Triangle::c};

            auto srcTriangle = &*src;
            auto triangle = std::move(*dst);
            auto dstTriangle = &triangle;
            if (direction[0]) {
                std::swap(srcTriangle, dstTriangle);
            }
            dstTriangle->*(vertices[d]) = srcTriangle->*(vertices[s]);
            if ((selector & 1) == 0) {
                if (direction[1]) {
                    std::swap(srcTriangle, dstTriangle);
                }
                dstTriangle->*(vertices[(d + 1) % 3]) = srcTriangle->*(vertices[(s + 1) % 3]);
            }
            break;
        }
        default: {
            assert(false);  // no no-op
        }
        }
    }

    void mutateBoxes(ptrdiff_t action)
    {
        assert(boxWorld);
        const auto getBox = [this] {
            assert(!std::empty(triangles));
            assert((std::size(triangles) % kBoxTriangleCount) == 0);
            return std::next(std::begin(triangles), kBoxTriangleCount * uniformInt(gen, UniformIntParam(0, int(std::size(triangles) / kBoxTriangleCount - 1))));
        };
        const auto sampleBoxVertex = [](auto box, const Vertex * anchor = nullptr) -> Vertex {
            const Vertex * vertices[] = {&box[0].a, &box[0].b, &box[2].b, &box[2].c, &box[4].b, &box[4].c, &box[5].c, &box[6].c};
            [[maybe_unused]] const auto vertexLess = [](auto l, auto r) { return *l < *r; };
            assert(std::adjacent_find(std::cbegin(vertices), std::cend(vertices), std::not_fn(vertexLess)) == std::cend(vertices));
            if (anchor) {
                std::shuffle(std::begin(vertices), std::end(vertices), gen);
                for (auto v : vertices) {
                    if (*anchor != *v) {
                        return *v;
                    }
                }
                INVARIANT(false);
            } else {
                const Vertex * v = nullptr;
                std::sample(std::cbegin(vertices), std::cend(vertices), &v, 1, gen);
                return *v;
            }
        };
        const auto numBoxes = std::size(triangles) / kBoxTriangleCount;
        switch (action) {
        case 0: {
            if (numBoxes > 1) {
                auto srcBox = getBox();
                auto dstBox = getBox();
                if (srcBox == dstBox) {
                    if (dstBox == std::begin(triangles)) {
                        dstBox = std::next(dstBox, kBoxTriangleCount);
                    } else {
                        dstBox = std::prev(dstBox, kBoxTriangleCount);
                    }
                }
                while ((srcBox[0].a == dstBox[0].a) || (std::next(srcBox, kBoxTriangleCount / 2)->c == std::next(dstBox, kBoxTriangleCount / 2)->c)) {
                    generateBox(dstBox);
                }
                auto srcVertex = sampleBoxVertex(srcBox);
                auto dstSaveVertex = sampleBoxVertex(dstBox, &srcVertex);
                std::tie(dstSaveVertex.x, srcVertex.x) = std::minmax({dstSaveVertex.x, srcVertex.x});
                std::tie(dstSaveVertex.y, srcVertex.y) = std::minmax({dstSaveVertex.y, srcVertex.y});
                std::tie(dstSaveVertex.z, srcVertex.z) = std::minmax({dstSaveVertex.z, srcVertex.z});
                [[maybe_unused]] auto boxEnd = putBox(dstBox, dstSaveVertex, srcVertex);
                assert(std::is_sorted(dstBox, boxEnd));
            } else {
            }
            break;
        }
        default: {
            assert(false);  // no no-op
        }
        }
    }

    template<size_t N>
    static constexpr auto cdf(float (&&probabilities)[N])
    {
        std::array<float, N> result;
        std::inclusive_scan(std::cbegin(probabilities), std::cend(probabilities), std::begin(result));
        return result;
    }

    void mutate()
    {
        if (boxWorld) {
            static const auto action = cdf({1.0f});
            const float probability = std::generate_canonical<float, std::numeric_limits<float>::digits>(gen);
            mutateBoxes(std::distance(std::cbegin(action), std::upper_bound(std::cbegin(action), std::cend(action), probability * action.back())));
        } else {
            static const auto action = cdf({0.1f, 0.1f, 0.1f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});
            const float probability = std::generate_canonical<float, std::numeric_limits<float>::digits>(gen);
            mutateTriangles(std::distance(std::cbegin(action), std::upper_bound(std::cbegin(action), std::cend(action), probability * action.back())));
        }
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
        if (params.maxDepth < testInput.params.maxDepth) {
            params.maxDepth = testInput.params.maxDepth;
        }

        triangles.reserve(std::size(triangles) + std::size(testInput.triangles));
        triangles.insert(std::cend(triangles), std::make_move_iterator(std::begin(testInput.triangles)), std::make_move_iterator(std::end(testInput.triangles)));
        assert(checkItems(std::cbegin(triangles), std::cend(triangles)));
    }
};

char ** findArg(char ** beg, char ** end, const char * arg)
{
    size_t argLen = std::strlen(arg);
    for (; beg != end; ++beg) {
        if (std::strncmp(*beg, arg, argLen) == 0) {
            return beg;
        }
    }
    return nullptr;
}

size_t readIntArg(char * arg, size_t argSize)
{
    size_t result = 0;
    auto argBeg = arg + argSize;
    auto argEnd = std::next(argBeg, std::strlen(arg + argSize));
    auto [p, ec] = std::from_chars(argBeg, argEnd, result);
    if ((ec != std::errc{}) || (p != argEnd)) {
        fmt::print(stderr, fg(fmt::color::red), "INFO(sah_kd_tree): cannot convert value '{}' of command line parameter {} to size_t\n", fmt::string_view{arg + argSize}, fmt::string_view{arg + 1, argSize - 2});
        std::exit(EXIT_FAILURE);
    }
    return result;
}

void writeIntArg(std::string & arg, size_t argSize, size_t argValue)
{
    arg.resize(argSize + std::numeric_limits<decltype(argValue)>::digits10 + 1);
    char * s = arg.data();
    auto [p, ec] = std::to_chars(std::next(s, argSize), std::next(s, arg.size()), argValue);
    if (ec != std::errc{}) {
        fmt::print(stderr, fg(fmt::color::red), "INFO(sah_kd_tree): cannot convert value '{}' of command line parameter {} from size_t to string\n", argValue, fmt::string_view{arg.data() + 1, argSize - 2});
        std::exit(EXIT_FAILURE);
    }
    assert(*p == '\0');
}
}  // namespace
}  // namespace fuzzer

extern "C"
{
    int LLVMFuzzerInitialize(int * argc, char *** argv)
    {
        static std::string maxLenOption = "-max_len=";
        static const size_t maxLenSize = maxLenOption.size();
        char ** maxLenArg = fuzzer::findArg(*argv + 1, *argv + *argc, maxLenOption.c_str());

        static std::string maxPrimitiveCountOption = "-max_primitive_count=";
        static const size_t maxPrimitiveCountSize = maxPrimitiveCountOption.size();
        char ** maxPrimitiveCountArg = fuzzer::findArg(*argv + 1, *argv + *argc, maxPrimitiveCountOption.c_str());

        static std::string boxWorldOption = "-box_world=";
        static const size_t boxWorldSize = boxWorldOption.size();
        char ** boxWorldArg = fuzzer::findArg(*argv + 1, *argv + *argc, boxWorldOption.c_str());

        if (boxWorldArg) {
            fuzzer::boxWorld = fuzzer::readIntArg(*boxWorldArg, boxWorldSize) != 0;
            fmt::print(stderr, "INFO(sah_kd_tree): generating of {} enabled\n", fuzzer::primitiveName());
        }

        if (!maxLenArg && !maxPrimitiveCountArg) {
            fmt::print(stderr, "INFO(sah_kd_tree): no primitive count limiting command line options are provided; number of {} is not limited\n", fuzzer::primitiveName());
            return 0;
        }

        if (maxLenArg && maxPrimitiveCountArg) {
            fmt::print(stderr, fg(fmt::color::red), "INFO(sah_kd_tree): max_len and max_primitive_count should not be set both at once\n");
            std::exit(EXIT_FAILURE);
        }

        if (maxLenArg) {
            size_t maxLen = fuzzer::readIntArg(*maxLenArg, maxLenSize);
            size_t itemCount = (std::max(maxLen, sizeof(fuzzer::Params)) - sizeof(fuzzer::Params)) / fuzzer::itemSize();
            fmt::print(stderr, "INFO(sah_kd_tree): maximum {} count: {}\n", fuzzer::primitiveName(), itemCount);
            if (itemCount == 0) {
                fmt::print(stderr, fg(fmt::color::red), "INFO(sah_kd_tree): nothing to fuzz\n");
                std::exit(EXIT_FAILURE);
            }
            return 0;
        }

        if (maxPrimitiveCountArg) {
            size_t maxPrimitiveCount = fuzzer::readIntArg(*maxPrimitiveCountArg, maxPrimitiveCountSize);
            fmt::print(stderr, "INFO(sah_kd_tree): maximum {} count: {}\n", fuzzer::primitiveName(), maxPrimitiveCount);
            if (maxPrimitiveCount == 0) {
                fmt::print(stderr, fg(fmt::color::red), "INFO(sah_kd_tree): nothing to fuzz\n");
                std::exit(EXIT_FAILURE);
            }
            size_t maxLen = sizeof(fuzzer::Params) + maxPrimitiveCount * fuzzer::itemSize();
            fuzzer::writeIntArg(maxLenOption, maxLenSize, maxLen);
            *maxPrimitiveCountArg = maxLenOption.data();
        } else {
            INVARIANT(false);
        }

        if (boxWorldArg) {
            *argv = std::rotate(*argv, boxWorldArg, std::next(boxWorldArg));
            --*argc;
        }
        return 0;
    }

    size_t LLVMFuzzerCustomMutator(uint8_t * data, size_t size, size_t maxSize, unsigned int seed)
    {
        fuzzer::setSeed(seed);

        fuzzer::TestInput testInput;
        if (testInput.read(data, size)) {
            testInput.mutate();
        } else {
            testInput.generate();
        }
        return testInput.write(data, maxSize);
    }

    size_t LLVMFuzzerCustomCrossOver(const uint8_t * data1, size_t size1, const uint8_t * data2, size_t size2, uint8_t * out, size_t maxOutSize, unsigned int seed)
    {
        fuzzer::setSeed(seed);

        fuzzer::TestInput testInput1;
        if (!testInput1.read(data1, size1)) {
            testInput1.generate();
        }
        fuzzer::TestInput testInput2;
        if (!testInput2.read(data2, size2)) {
            testInput2.generate();
        }
        testInput1.cross(testInput2);
        return testInput1.write(out, maxOutSize);
    }

    int LLVMFuzzerTestOneInput(const uint8_t * data, size_t size)
    {
        fuzzer::TestInput testInput;
        if (!testInput.read(data, size)) {
            return 0;
        }

        fuzzer::testOneInput(testInput.params, testInput.triangles);

        return 0;
    }
}  // extern "C"

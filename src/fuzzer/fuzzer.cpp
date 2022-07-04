#include <sah_kd_tree/helpers/setup.cuh>
#include <sah_kd_tree/sah_kd_tree.cuh>
#include <sah_kd_tree/types.cuh>

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
#include <optional>
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

#define INVARIANT(condition)            \
    {                                   \
        if (!(condition)) std::abort(); \
    }

using sah_kd_tree::F;
using sah_kd_tree::I;
using sah_kd_tree::U;

struct Vertex
{
    F x, y, z;

    bool operator<(const Vertex & rhs) const
    {
        return std::tie(x, y, z) < std::tie(rhs.x, rhs.y, rhs.z);
    }

    bool operator==(const Vertex & rhs) const  // legal for non-calculated floating-point values
    {
        return std::tie(x, y, z) == std::tie(rhs.x, rhs.y, rhs.z);
    }
};

struct Triangle
{
    Vertex a, b, c;

    bool operator<(const Triangle & rhs) const
    {
        return std::tie(a, b, c) < std::tie(rhs.a, rhs.b, rhs.c);
    }

    bool operator==(const Triangle & rhs) const
    {
        return std::tie(a, b, c) == std::tie(rhs.a, rhs.b, rhs.c);
    }
};

namespace
{

constexpr bool kBoxWorld = false;
constexpr size_t kBoxTriangleCount = 12;
constexpr size_t kMaxTriangleCount = kBoxTriangleCount * 6;  // 2082 low poly deer
static_assert(!kBoxWorld || ((kMaxTriangleCount % kBoxTriangleCount) == 0));

constexpr bool kSortItems = false;

constexpr int kIntBboxSize = 10;
constexpr bool kFuzzIntegerCoordinate = false;

constexpr int kFloatDigits = std::numeric_limits<F>::digits;

using RandomValueType = typename std::mt19937::result_type;
using UniformIntDistribution = std::uniform_int_distribution<>;
using UniformIntParam = typename UniformIntDistribution::param_type;

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

void genComponent(F & f, int min = -kIntBboxSize, int max = +kIntBboxSize)
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

Vertex genVertex()
{
    Vertex vertex;
    genVertex(vertex);
    return vertex;
}

void genTriangle(Triangle & triangle)
{
    auto & [a, b, c] = triangle;
    genVertex(a);
    genVertex(b);
    genVertex(c);

    if (kSortItems) {
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

Triangle genTriangle()
{
    Triangle triangle;
    genTriangle(triangle);
    return triangle;
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
        min = genVertex();
        max = genVertex();
        std::tie(min.x, max.x) = std::minmax({min.x, max.x});
        std::tie(min.y, max.y) = std::minmax({min.y, max.y});
        std::tie(min.z, max.z) = std::minmax({min.z, max.z});
    } while (!(min.x < max.x) || !(min.y < max.y) || !(min.z < max.z));
    putBox(out, min, max);
    return out;
}

template<typename TriangleRandomAccessIterator>
void sortItems(TriangleRandomAccessIterator beg, TriangleRandomAccessIterator end)
{
    if (kBoxWorld) {
        assert((std::distance(beg, end) % kBoxTriangleCount) == 0);
#ifndef NDEBUG
        for (auto box = beg; box != end; std::advance(box, kBoxTriangleCount)) {
            assert(std::is_sorted(box, std::next(box, kBoxTriangleCount)));
        }
#endif
        if (kSortItems) {
            size_t boxCount = size_t(std::distance(beg, end)) / kBoxTriangleCount;
            std::vector<size_t> boxes(boxCount);
            std::iota(std::begin(boxes), std::end(boxes), 0);
            const auto boxLess = [beg](size_t lhs, size_t rhs) -> bool {
                auto lhsBeg = std::next(beg, lhs * kBoxTriangleCount);
                auto lhsEnd = std::next(lhsBeg, kBoxTriangleCount);
                auto rhsBeg = std::next(beg, rhs * kBoxTriangleCount);
                auto rhsEnd = std::next(rhsBeg, kBoxTriangleCount);
                return std::lexicographical_compare(lhsBeg, lhsEnd, rhsBeg, rhsEnd);
            };
            std::sort(std::begin(boxes), std::end(boxes), boxLess);
            for (size_t i = 0; i < std::size(boxes); ++i) {
                for (size_t j = i; j != boxes[j]; std::swap(j, boxes[j])) {
                    auto lhsBeg = std::next(beg, j * kBoxTriangleCount);
                    auto lhsEnd = std::next(lhsBeg, kBoxTriangleCount);
                    auto rhsBeg = std::next(beg, boxes[j] * kBoxTriangleCount);
                    std::swap_ranges(lhsBeg, lhsEnd, rhsBeg);
                }
            }
        }
    } else {
        if (kSortItems) {
            std::sort(beg, end);
        }
    }
}

template<typename TriangleRandomAccessIterator>
bool checkItems(TriangleRandomAccessIterator beg, TriangleRandomAccessIterator end)
{
    if (kBoxWorld) {
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
        if (kSortItems) {
            size_t boxCount = size_t(std::distance(beg, end)) / kBoxTriangleCount;
            std::vector<size_t> boxes(boxCount);
            std::iota(std::begin(boxes), std::end(boxes), 0);
            const auto boxLess = [beg](size_t lhs, size_t rhs) -> bool {
                auto lhsBeg = std::next(beg, lhs * kBoxTriangleCount);
                auto lhsEnd = std::next(lhsBeg, kBoxTriangleCount);
                auto rhsBeg = std::next(beg, rhs * kBoxTriangleCount);
                auto rhsEnd = std::next(rhsBeg, kBoxTriangleCount);
                return std::lexicographical_compare(lhsBeg, lhsEnd, rhsBeg, rhsEnd);
            };
            if (!std::is_sorted(std::begin(boxes), std::end(boxes), boxLess)) {
                return false;
            }
        }
    } else {
        if (kSortItems) {
            if (!std::is_sorted(beg, end)) {
                return false;
            }
        }
    }
    return true;
}

struct TestInput
{
    static constexpr auto kTrianglesPerItem = kBoxWorld ? kBoxTriangleCount : 1;
    static constexpr auto kItemSize = sizeof(Triangle) * kTrianglesPerItem;

    sah_kd_tree::Params params;
    std::vector<Triangle> triangles;

    static_assert(std::is_standard_layout_v<sah_kd_tree::Params> && std::is_trivially_copyable_v<sah_kd_tree::Params>, "!");
    static_assert(std::is_standard_layout_v<Triangle> && std::is_trivially_copyable_v<Triangle>, "!");

    void generate(size_t triangleCount = kTrianglesPerItem)
    {
        assert(0 < triangleCount);
        assert((triangleCount % kTrianglesPerItem) == 0);
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
                assert(!kBoxWorld);
                Vertex v[4];
                for (Vertex & vertex : v) {
                    genVertex(vertex);
                }
                triangles.push_back({v[1], v[2], v[3]});
                triangles.push_back({v[2], v[3], v[0]});
                triangles.push_back({v[3], v[0], v[1]});
                triangles.push_back({v[0], v[1], v[2]});
            } else {
                assert(!kBoxWorld);
                triangles.emplace_back(genTriangle());
            }
        }
        sortItems(std::begin(triangles), std::end(triangles));
    }

    bool read(const uint8_t * data, size_t size)
    {
        if (size < sizeof(sah_kd_tree::Params) + kItemSize) {
            return false;
        }
        if (((size - sizeof(sah_kd_tree::Params)) % kItemSize) != 0) {
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
        assert(!(maxSize < sizeof(Params) + kItemSize));
        assert(checkItems(std::cbegin(triangles), std::cend(triangles)));

        size_t size = 0;

        std::memcpy(data, &params, sizeof params);
        data += sizeof params;
        size += sizeof params;

        if (const size_t maxItemCount = (maxSize - size) / kItemSize; maxItemCount < std::size(triangles) / kTrianglesPerItem) {
            std::vector<typename decltype(triangles)::const_iterator> servived;
            servived.reserve(std::size(triangles) / kTrianglesPerItem);
            for (auto t = std::begin(triangles); t != std::end(triangles); std::advance(t, kTrianglesPerItem)) {
                servived.push_back(t);
            }
            std::shuffle(std::begin(servived), std::end(servived), gen);
            servived.resize(maxItemCount);
            if (kSortItems) {
                std::sort(std::begin(servived), std::end(servived));
            }
            for (auto t : servived) {
                std::memcpy(data, &*t, kItemSize);
                data += kItemSize;
                size += kItemSize;
            }
            assert(((size - sizeof params) % kItemSize) == 0);
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
        assert(!kBoxWorld);
        if (kSortItems) {
            assert(std::is_sorted(std::cbegin(triangles), std::cend(triangles)));
        }
        const auto getTriangle = [this] {
            assert(!std::empty(triangles));
            return std::next(std::begin(triangles), uniformInt(gen, UniformIntParam(0, int(std::size(triangles) - 1))));
        };
        const auto adjustTriangle = [](auto beg, auto mid, auto end) {
            if ((mid != beg) && (*mid < *std::prev(mid))) {
                return std::rotate(std::upper_bound(beg, mid, *mid), mid, std::next(mid));
            } else if ((std::next(mid) != end) && (*std::next(mid) < *mid)) {
                return std::rotate(mid, std::next(mid), std::upper_bound(std::next(mid), end, *mid));
            } else {
                return mid;
            }
        };
        const auto mutateTriangle = [&](auto t) {
            genTriangle(*t);
            if (kSortItems) {
                adjustTriangle(std::begin(triangles), t, std::end(triangles));
                assert(std::is_sorted(std::cbegin(triangles), std::cend(triangles)));
            }
        };
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
                if (kSortItems) {
                    triangles.erase(t);
                } else {
                    std::iter_swap(t, std::prev(std::end(triangles)));
                    triangles.pop_back();
                }
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
            if (kSortItems) {
                adjustTriangle(std::begin(triangles), t, std::end(triangles));
                assert(std::is_sorted(std::cbegin(triangles), std::cend(triangles)));
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

            if (kSortItems) {
                std::sort(std::begin(triangles), std::end(triangles));
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
        assert(kBoxWorld);
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
                    if (((v->x < anchor->x) || (anchor->x < v->x)) && ((v->y < anchor->y) || (anchor->y < v->y)) && ((v->z < anchor->z) || (anchor->z < v->z))) {
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
    static constexpr auto cdf(float(&&probabilities)[N])
    {
        std::array<float, N> result;
        std::inclusive_scan(std::cbegin(probabilities), std::cend(probabilities), std::begin(result));
        return result;
    }

    void mutate()
    {
        if (kBoxWorld) {
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

        decltype(triangles) allTriangles;
        if (kBoxWorld) {
            allTriangles.reserve((std::size(triangles) + std::size(testInput.triangles)));
            std::vector<typename decltype(triangles)::const_iterator> mergedItems;
            mergedItems.reserve((std::size(triangles) + std::size(testInput.triangles)) / kTrianglesPerItem);
            for (auto t = std::begin(triangles); t != std::end(triangles); std::advance(t, kTrianglesPerItem)) {
                mergedItems.push_back(t);
            }
            for (auto t = std::begin(testInput.triangles); t != std::end(testInput.triangles); std::advance(t, kTrianglesPerItem)) {
                mergedItems.push_back(t);
            }
            const auto itemLess = [](auto lhs, auto rhs) -> bool { return std::lexicographical_compare(lhs, std::next(lhs, kTrianglesPerItem), rhs, std::next(rhs, kTrianglesPerItem)); };
            std::sort(std::begin(mergedItems), std::end(mergedItems), itemLess);
            auto out = std::back_inserter(allTriangles);
            for (auto item : mergedItems) {
                out = std::move(item, std::next(item, kTrianglesPerItem), out);
            }
        } else {
            allTriangles.resize((std::size(triangles) + std::size(testInput.triangles)));
            [[maybe_unused]] const auto trianglesEnd = std::merge(std::cbegin(triangles), std::cend(triangles), std::cbegin(testInput.triangles), std::cend(testInput.triangles), std::begin(allTriangles));
            assert(trianglesEnd == std::end(allTriangles));
        }
        std::swap(triangles, allTriangles);
        assert(checkItems(std::cbegin(triangles), std::cend(triangles)));
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
        if (size_t(*argc) + (std::extent_v<decltype(options)>) > std::extent_v<decltype(args)>) {
            std::exit(EXIT_FAILURE);
        }

        const auto maxLen = sizeof(sah_kd_tree::Params) + kMaxTriangleCount * sizeof(Triangle);
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
        setSeed(seed);

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
        setSeed(seed);

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

        sah_kd_tree::helpers::Triangles triangles;
        sah_kd_tree::helpers::setTriangles(triangles, std::cbegin(testInput.triangles), std::cend(testInput.triangles));

        sah_kd_tree::Builder builder;
        sah_kd_tree::helpers::linkTriangles(builder, triangles);

        sah_kd_tree::Tree tree = builder(testInput.params);
        if (std::size(testInput.triangles) < tree.depth) {
            std::abort();
        }
        return 0;
    }
}

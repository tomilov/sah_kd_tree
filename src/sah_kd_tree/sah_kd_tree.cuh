#pragma once

#include <thrust/device_allocator.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/pair.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <fmt/chrono.h>
#include <spdlog/spdlog.h>

#include <chrono>
#include <functional>
#include <limits>
#include <source_location>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>

#include <sah_kd_tree/sah_kd_tree_export.h>

namespace sah_kd_tree
{

template<
    typename U,
    typename T>
U safeConvert(T size)
{
    if (!std::in_range<U>(size)) {
        throw std::range_error{"safeConvert"};
    }
    return static_cast<U>(size);
}

class ScopeTimer
{
public:
    using Clock = std::chrono::high_resolution_clock;

    explicit ScopeTimer(const std::source_location & sourceLocationIn = std::source_location::current())
        : sourceLocation{sourceLocationIn}
    {}

    ~ScopeTimer()
    {
        auto dt = std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - start);
        SPDLOG_TRACE("{}:{}:{} in {}: {}", sourceLocation.file_name(), sourceLocation.line(), sourceLocation.column(), sourceLocation.function_name(), dt);
    }

private:
    const std::source_location sourceLocation;
    const Clock::time_point start = Clock::now();
};

struct DefaultTraits
{
    using I = std::int32_t;
    using U = std::uint32_t;
    using F = float;
    template<typename T>
    using Allocator = thrust::device_allocator<T>;
    template<typename T>
    using Vector = thrust::device_vector<T, Allocator<T>>;
    using Exec = decltype(thrust::device);
    using Progress = std::function<bool(size_t progressValue)>;
    using ComponentIterator = thrust::permutation_iterator<typename Vector<F>::const_iterator, typename Vector<U>::const_iterator>;
};

template<typename Traits = DefaultTraits>
struct Params
{
    using I = typename Traits::I;
    using U = typename Traits::U;
    using F = typename Traits::F;
    template<typename T>
    using Vector = typename Traits::template Vector<T>;

    F emptinessFactor = 0.8f;   // (0, 1]
    F traversalCost = 2.0f;     // (0, inf)
    F intersectionCost = 1.0f;  // (0, inf)
    U maxTreeDepth = std::numeric_limits<U>::max();
};

template<typename Traits = DefaultTraits>
struct Tree
{
    using I = typename Traits::I;
    using U = typename Traits::U;
    using F = typename Traits::F;
    template<typename T>
    using Allocator = typename Traits::template Allocator<T>;
    template<typename T>
    using Vector = typename Traits::template Vector<T>;

    [[no_unique_address]] Allocator<std::byte> allocator;

    thrust::host_vector<U> layerDepth;

    struct Projection
    {
        [[no_unique_address]] Allocator<std::byte> allocator;

        struct Node
        {
            [[no_unique_address]] Allocator<std::byte> allocator;

            Vector<F> min{allocator}, max{allocator};
            Vector<U> leftRope{allocator}, rightRope{allocator};
        } node{allocator};
    } x{allocator}, y{allocator}, z{allocator};

    struct Node
    {
        [[no_unique_address]] Allocator<std::byte> allocator;

        Vector<I> splitDimension{allocator};
        Vector<F> splitPos{allocator};
        Vector<U> leftChild{allocator}, rightChild{allocator};
        Vector<U> parent{allocator};
    } node{allocator};

    Vector<U> polygonTriangle{allocator};

    Tree() = default;
    Tree(Tree &) = delete;
    Tree & operator=(Tree &) = delete;

    Tree(const Allocator<std::byte> & allocatorIn)
        : allocator{allocatorIn}
    {}
};

template<typename Traits = DefaultTraits>
struct Projection
{
    using I = typename Traits::I;
    using U = typename Traits::U;
    using F = typename Traits::F;
    template<typename T>
    using Allocator = typename Traits::template Allocator<T>;
    using Exec = typename Traits::Exec;
    template<typename T>
    using Vector = typename Traits::template Vector<T>;
    using ComponentIterator = typename Traits::ComponentIterator;

    [[no_unique_address]] Allocator<std::byte> allocator;
    [[no_unique_address]] Exec exec;

    struct ToPair
    {
        __host__ __device__ cuda::std::pair<
            U,
            U>
        operator()(U value) const
        {
            return {value, value};
        }
    } toPair;

    struct ToEventPos
    {
        __host__ __device__ F operator()(
            I eventKind,
            cuda::std::tuple<
                F,
                F> bbox) const
        {
            return (eventKind < 0) ? thrust::get<1>(bbox) : thrust::get<0>(bbox);
        }
    } toEventPos;

    struct Triangle
    {
        U count = 0;
        ComponentIterator a, b, c;
    } triangle;

    struct Polygon
    {
        [[no_unique_address]] Allocator<std::byte> allocator;

        Vector<F> min{allocator}, max{allocator};
    } polygon{allocator};

    struct Node
    {
        [[no_unique_address]] Allocator<std::byte> allocator;

        Vector<F> min{allocator}, max{allocator};
        Vector<U> leftRope{allocator}, rightRope{allocator};
    } node{allocator};

    struct Event
    {
        [[no_unique_address]] Allocator<std::byte> allocator;

        U count = 0;
        Vector<U> node{allocator};
        Vector<F> pos{allocator};
        Vector<I> kind{allocator};  // TODO: scale event kind by polygon value
        Vector<U> polygon{allocator};

        Vector<U> polygonCountLeft{allocator}, polygonCountRight{allocator};  // or eventLeft, eventRight mutually exclusive
    } event{allocator};

    struct Layer
    {
        [[no_unique_address]] Allocator<std::byte> allocator;

        Vector<F> splitCost{allocator};
        Vector<U> splitEvent{allocator};
        Vector<F> splitPos{allocator};

        Vector<U> polygonCountLeft{allocator}, polygonCountRight{allocator};
        Vector<U> splittedPolygonCount{allocator};  // can be optimized out
    } layer{allocator};

    Projection() = default;
    Projection(Projection &) = delete;
    Projection & operator=(Projection &) = delete;

    Projection(const Allocator<std::byte> & allocatorIn)
        : allocator{allocatorIn}
    {}

    Projection(Exec execIn)
        : exec{execIn}
    {}

    Projection(
        const Allocator<std::byte> & allocatorIn,
        Exec execIn)
        : allocator{allocatorIn}
        , exec{execIn}
    {}

    void calculateTriangleBbox();
    void calculateRootNodeBbox();
    void generateInitialEvent();

    void findPerfectSplit(
        const Params<Traits> & sah,
        U layerSize,
        const Vector<U> & layerNodeOffset,
        const Vector<U> & nodePolygonCount,
        const Projection & y,
        const Projection & z);
    void decoupleEventBoth(
        const Vector<I> & nodeSplitDimension,
        const Vector<I> & polygonSide);

    void mergeEvent(
        U polygonCount,
        U splittedPolygonCount,
        const Vector<U> & polygonNode,
        const Vector<U> & splittedPolygon);
};

template<typename Traits = DefaultTraits>
struct Builder
{
    using I = typename Traits::I;
    using U = typename Traits::U;
    using F = typename Traits::F;
    template<typename T>
    using Allocator = typename Traits::template Allocator<T>;
    using Exec = typename Traits::Exec;
    template<typename T>
    using Vector = typename Traits::template Vector<T>;
    using Progress = typename Traits::Progress;

    static inline constexpr I kNoSplitDimension = -1;  // leaf node

    struct IsNotLeaf
    {
        __host__ __device__ bool operator()(I nodeSplitDimension) const
        {
            return !(nodeSplitDimension < 0);
        }
    } isNotLeaf;

    struct IsNodeNotEmpty
    {
        __host__ __device__ bool operator()(U nodePolygonCount) const
        {
            return nodePolygonCount != 0;
        }
    } isNodeNotEmpty;

    [[no_unique_address]] Allocator<std::byte> allocator;
    [[no_unique_address]] Exec exec;

    struct Polygon
    {
        [[no_unique_address]] Allocator<std::byte> allocator;

        U count = 0;
        U splittedCount = 0;

        Vector<U> triangle{allocator};
        Vector<U> node{allocator};
        Vector<I> side{allocator};
        Vector<U> eventRight{allocator};  // right event in diverse best dimensions
    } polygon{allocator};

    struct Node
    {
        [[no_unique_address]] Allocator<std::byte> allocator;

        U count = 1;  // always equal layer.base + layer.size

        Vector<I> splitDimension{allocator};
        Vector<F> splitPos{allocator};                                                                 // TODO: splitDimension can be packed into 2 lsb of splitPos
        Vector<U> leftChild{allocator}, rightChild{allocator};                                         // left child node and right child node if not leaf, polygon range otherwise
        Vector<U> polygonCount{allocator}, polygonCountLeft{allocator}, polygonCountRight{allocator};  // unique polygon count in the current node, in its left child node and in its right child node correspondingly
        Vector<U> parent{allocator};                                                                   // temporarily needed to build ropes
    } node{allocator};                                                                                 // TODO: optimize out node.rightChild

    struct Leaf
    {
        [[no_unique_address]] Allocator<std::byte> allocator;

        U count = 0;

        Vector<U> node{allocator};
        Vector<U> polygonCount{allocator};
        Vector<U> polygonOffset{allocator};
    } leaf{allocator};

    struct Layer
    {
        [[no_unique_address]] Allocator<std::byte> allocator;

        U base = 0;
        U size = 1;

        Vector<U> nodeOffset{allocator};
    } layer{allocator};

    Vector<U> splittedPolygon{allocator};

    Builder() = default;
    Builder(Builder &) = delete;
    Builder & operator=(Builder &) = delete;

    Builder(const Allocator<std::byte> & allocatorIn)
        : allocator{allocatorIn}
    {}

    Builder(Exec execIn)
        : exec{execIn}
    {}

    Builder(
        const Allocator<std::byte> & allocatorIn,
        Exec execIn)
        : allocator{allocatorIn}
        , exec{execIn}
    {}

    void filterLayerNodeOffset();
    void selectNodeBestSplit(
        const Params<Traits> & sah,
        const Projection<Traits> & x,
        const Projection<Traits> & y,
        const Projection<Traits> & z);
    template<I dimension>
    void determinePolygonSide(const Projection<Traits> & projection);
    void updateSplittedPolygonCount();
    void separateSplittedPolygon();
    void updatePolygonNode();
    template<I dimension>
    void splitPolygon(
        Projection<Traits> & x,
        const Projection<Traits> & y,
        const Projection<Traits> & z) const;
    void updateSplittedPolygonNode();
    void setNodeCount(
        Projection<Traits> & x,
        Projection<Traits> & y,
        Projection<Traits> & z) const;
    template<I dimension>
    void splitNode(
        U layerBasePrev,
        Projection<Traits> & projection) const;
    void resizeNode();
    void populateNodeParent();
    void populateLeafNodeTriangleRange();

    bool checkBoxes(
        const Projection<Traits> & x,
        const Projection<Traits> & y,
        const Projection<Traits> & z) const;
    bool checkNodes(
        const Projection<Traits> & x,
        const Projection<Traits> & y,
        const Projection<Traits> & z) const;

    template<
        I dimension,
        bool forth>
    void calculateRope(
        Projection<Traits> & x,
        const Projection<Traits> & y,
        const Projection<Traits> & z) const;

    template<typename P = Progress>
    bool build(
        const P & progress,
        const Params<Traits> & sah,
        Projection<Traits> & x,
        Projection<Traits> & y,
        Projection<Traits> & z,
        Tree<Traits> & tree);
};

extern template bool Builder<>::build<>(
    const Progress & progress,
    const Params<> & sah,
    Projection<> & x,
    Projection<> & y,
    Projection<> & z,
    Tree<> & tree) SAH_KD_TREE_EXPORT;

}  // namespace sah_kd_tree

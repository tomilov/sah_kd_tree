#pragma once

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/pair.h>
#include <thrust/tuple.h>

#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <type_traits>

#include <cassert>
#include <cstddef>

#include <sah_kd_tree/sah_kd_tree_export.h>

namespace sah_kd_tree
{

template<typename U>
U sizeToU(size_t size)
{
    assert(size <= std::numeric_limits<U>::max());
    return static_cast<U>(size);
}

struct DefaultTraits
{
    using I = int;
    using U = unsigned int;
    using F = float;
    template<typename T>
    using Allocator = thrust::device_allocator<T>;
    template<typename T>
    using Vector = thrust::device_vector<T, Allocator<T>>;
    using Progress = std::function<bool(float progressValue, const std::string & progressText)>;
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
    U maxDepth = std::numeric_limits<U>::max();
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

    Allocator<void> allocator;

    thrust::host_vector<U> layerDepth;

    struct Projection
    {
        struct Node
        {
            Vector<F> min, max;
            Vector<U> leftRope, rightRope;
        } node;
    } x, y, z;

    struct Polygon
    {
        Vector<U> triangle;
    } polygon;

    struct Node
    {
        Vector<I> splitDimension;
        Vector<F> splitPos;
        Vector<U> leftChild, rightChild;
        Vector<U> parent;
    } node;

    Tree() = default;

    Tree(const Allocator<void> & allocator)
        : allocator{allocator}
        , x{
              .node{
                  .min{allocator},
                  .max{allocator},
                  .leftRope{allocator},
                  .rightRope{allocator},
              },
          }
        , y{
              .node{
                  .min{allocator},
                  .max{allocator},
                  .leftRope{allocator},
                  .rightRope{allocator},
              },
          }
        , z{
              .node{
                  .min{allocator},
                  .max{allocator},
                  .leftRope{allocator},
                  .rightRope{allocator},
              },
          }
        , polygon{
              .triangle{allocator}
          }
        , node{
              .splitDimension{allocator},
              .splitPos{allocator},
              .leftChild{allocator},
              .rightChild{allocator},
              .parent{allocator},
          }
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
    template<typename T>
    using Vector = typename Traits::template Vector<T>;

    Allocator<void> allocator;

    struct ToPair
    {
        __host__ __device__ thrust::pair<U, U> operator()(U value) const
        {
            return {value, value};
        }
    } toPair;

    struct ToEventPos
    {
        __host__ __device__ F operator()(I eventKind, thrust::tuple<F, F> bbox) const
        {
            return (eventKind < 0) ? thrust::get<1>(bbox) : thrust::get<0>(bbox);
        }
    } toEventPos;

    struct Triangle
    {
        U count = 0;
        typename Vector<F>::const_pointer a, b, c;
    } triangle;

    struct Polygon
    {
        Vector<F> min, max;
    } polygon;

    struct Node
    {
        Vector<F> min, max;
        Vector<U> leftRope, rightRope;
    } node;

    struct Event
    {
        U count = 0;
        Vector<U> node;
        Vector<F> pos;
        Vector<I> kind;  // TODO: scale event kind by polygon value
        Vector<U> polygon;

        Vector<U> polygonCountLeft, polygonCountRight;  // or eventLeft, eventRight mutually exclusive
    } event;

    struct Layer
    {
        Vector<F> splitCost;
        Vector<U> splitEvent;
        Vector<F> splitPos;

        Vector<U> polygonCountLeft, polygonCountRight;
        Vector<U> splittedPolygonCount;  // can be optimized out
    } layer;

    Projection() = default;

    Projection(const Allocator<void> & allocator)
        : allocator{allocator}
        , polygon{
              .min{allocator},
              .max{allocator},
          }
        , node{
              .min{allocator},
              .max{allocator},
              .leftRope{allocator},
              .rightRope{allocator},
          }
        , event{
              .node{allocator},
              .pos{allocator},
              .kind{allocator},
              .polygon{allocator},
              .polygonCountLeft{allocator},
              .polygonCountRight{allocator},
          }
        , layer{
              .splitCost{allocator},
              .splitEvent{allocator},
              .splitPos{allocator},
              .polygonCountLeft{allocator},
              .polygonCountRight{allocator},
              .splittedPolygonCount{allocator},
          }
    {}

    void calculateTriangleBbox();
    void calculateRootNodeBbox();
    void generateInitialEvent();

    void findPerfectSplit(const Params<Traits> & sah, U layerSize, const Vector<U> & layerNodeOffset, const Vector<U> & nodePolygonCount, const Projection & y, const Projection & z);
    void decoupleEventBoth(const Vector<I> & nodeSplitDimension, const Vector<I> & polygonSide);

    void mergeEvent(U polygonCount, U splittedPolygonCount, const Vector<U> & polygonNode, const Vector<U> & splittedPolygon);
};

template<typename Traits = DefaultTraits>
struct Builder
{
    using I = typename Traits::I;
    using U = typename Traits::U;
    using F = typename Traits::F;
    template<typename T>
    using Allocator = typename Traits::template Allocator<T>;
    template<typename T>
    using Vector = typename Traits::template Vector<T>;
    using Progress = typename Traits::Progress;

    static inline constexpr I kNoSplitDimension = -1;

    Allocator<void> allocator;

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

    struct Polygon
    {
        U count = 0;
        U splittedCount = 0;

        Vector<U> triangle;
        Vector<U> node;
        Vector<I> side;
        Vector<U> eventRight;  // right event in diverse best dimensions
    } polygon;

    struct Node
    {
        U count = 1;  // always equal layer.base + layer.size

        Vector<I> splitDimension;
        Vector<F> splitPos;                                           // TODO: splitDimension can be packed into 2 lsb of splitPos
        Vector<U> leftChild, rightChild;                              // left child node and right child node if not leaf, polygon range otherwise
        Vector<U> polygonCount, polygonCountLeft, polygonCountRight;  // unique polygon count in the current node, in its left child node and in its right child node correspondingly
        Vector<U> parent;                                             // temporarily needed to build ropes
    } node;                                                           // TODO: optimize out node.rightChild

    struct Leaf
    {
        U count = 0;

        Vector<U> node;
        Vector<U> polygonCount;
        Vector<U> polygonOffset;
    } leaf;

    struct Layer
    {
        U base = 0;
        U size = 1;

        Vector<U> nodeOffset;
    } layer;

    Vector<U> splittedPolygon;

    Builder() = default;

    Builder(const Allocator<void> & allocator)
        : allocator{allocator}
        , polygon{
              .triangle{allocator},
              .node{allocator},
              .side{allocator},
              .eventRight{allocator},
          }
        , node{
              .splitDimension{allocator},
              .splitPos{allocator},
              .leftChild{allocator},
              .rightChild{allocator},
              .polygonCount{allocator},
              .polygonCountLeft{allocator},
              .polygonCountRight{allocator},
              .parent{allocator},
          }
        , leaf{
              .node{allocator},
              .polygonCount{allocator},
              .polygonOffset{allocator},
          }
        , layer{
              .nodeOffset{allocator},
          }
        , splittedPolygon{allocator}
    {}

    void filterLayerNodeOffset();
    void selectNodeBestSplit(const Params<Traits> & sah, const Projection<Traits> & x, const Projection<Traits> & y, const Projection<Traits> & z);
    template<I dimension>
    void determinePolygonSide(const Projection<Traits> & projection);
    void updateSplittedPolygonCount();
    void separateSplittedPolygon();
    void updatePolygonNode();
    template<I dimension>
    void splitPolygon(Projection<Traits> & x, const Projection<Traits> & y, const Projection<Traits> & z) const;
    void updateSplittedPolygonNode();
    void setNodeCount(Projection<Traits> & x, Projection<Traits> & y, Projection<Traits> & z) const;
    template<I dimension>
    void splitNode(U layerBasePrev, Projection<Traits> & projection) const;
    void resizeNode();
    void populateNodeParent();
    void populateLeafNodeTriangleRange();

    bool checkTree(const Projection<Traits> & x, const Projection<Traits> & y, const Projection<Traits> & z) const;

    template<I dimension, bool forth>
    void calculateRope(Projection<Traits> & x, const Projection<Traits> & y, const Projection<Traits> & z) const;

    template<typename P = Progress>
    std::optional<Tree<Traits>> build(const P & progress, const Params<Traits> & sah, Projection<Traits> & x, Projection<Traits> & y, Projection<Traits> & z) SAH_KD_TREE_EXPORT;
};

template<typename Traits = DefaultTraits>
struct Triangle
{
    using I = typename Traits::I;
    using U = typename Traits::U;
    using F = typename Traits::F;
    template<typename T>
    using Allocator = typename Traits::template Allocator<T>;
    template<typename T>
    using Vector = typename Traits::template Vector<T>;

    Allocator<void> allocator;

    template<typename TriangleType, typename TransposedTriangleType>
    struct TransposeTriangle
    {
        __host__ __device__ TransposedTriangleType operator()(const TriangleType & t) const
        {
            return {{t.a.x, t.b.x, t.c.x}, {t.a.y, t.b.y, t.c.y}, {t.a.z, t.b.z, t.c.z}};
        }
    };

    U count = 0;

    struct Component
    {
        Vector<F> a, b, c;
    } x, y, z;

    Triangle() = default;

    Triangle(const Allocator<void> & allocator)
        : allocator{allocator}
        , x{
              .a{allocator},
              .b{allocator},
              .c{allocator},
          }
        , y{
              .a{allocator},
              .b{allocator},
              .c{allocator},
          }
        , z{
              .a{allocator},
              .b{allocator},
              .c{allocator},
          }
    {}

    // For non-CUDA THRUST_DEVICE_SYSTEM a using of the function works fine in pure .cpp,
    // but to conduct with .cpp code in case of CUDA THRUST_DEVICE_SYSTEM
    // a "glue" .hpp+.cu pair is required (ideally .hpp should contain only C++).
    // Even so there is a bug in CUDA:
    // https://forums.developer.nvidia.com/t/cuda-separable-compilation-shared-libraries-invalid-function-error/188476
    // Thus dlink the library only once or use static linking.
    template<typename TriangleIterator>
    void setTriangle(TriangleIterator triangleBegin, TriangleIterator triangleEnd)
    {
        using TriangleType = std::remove_const_t<thrust::iterator_value_t<TriangleIterator>>;
        thrust::device_vector<TriangleType, Allocator<TriangleType>> t{allocator};
        t.assign(triangleBegin, triangleEnd);
        count = sizeToU<U>(t.size());
        const auto transposeComponent = [this](typename Triangle::Component & component)
        {
            component.a.resize(count);
            component.b.resize(count);
            component.c.resize(count);
            return thrust::make_zip_iterator(component.a.begin(), component.b.begin(), component.c.begin());
        };
        auto transposedTriangleBegin = thrust::make_zip_iterator(transposeComponent(x), transposeComponent(y), transposeComponent(z));
        using TransposedTriangleType = thrust::iterator_value_t<decltype(transposedTriangleBegin)>;
        thrust::transform(t.cbegin(), t.cend(), transposedTriangleBegin, TransposeTriangle<TriangleType, TransposedTriangleType>{});
    }
};

template<typename Traits = DefaultTraits>
void linkTriangles(const Triangle<Traits> & triangle, Projection<Traits> & x, Projection<Traits> & y, Projection<Traits> & z, Builder<Traits> & builder) SAH_KD_TREE_EXPORT;

}  // namespace sah_kd_tree

#if SAH_KD_TREE_HEADER_ONLY
#include <sah_kd_tree/builder.inl>
#include <sah_kd_tree/calculate_root_node_bbox.inl>
#include <sah_kd_tree/calculate_rope.inl>
#include <sah_kd_tree/calculate_triangle_bbox.inl>
#include <sah_kd_tree/check_tree.inl>
#include <sah_kd_tree/decouple_event_both.inl>
#include <sah_kd_tree/determine_polygon_side.inl>
#include <sah_kd_tree/filter_layer_node_offset.inl>
#include <sah_kd_tree/find_perfect_split.inl>
#include <sah_kd_tree/generate_initial_event.inl>
#include <sah_kd_tree/link_triangle.inl>
#include <sah_kd_tree/merge_event.inl>
#include <sah_kd_tree/populate_leaf_node_triangle_range.inl>
#include <sah_kd_tree/populate_node_parent.inl>
#include <sah_kd_tree/resize_node.inl>
#include <sah_kd_tree/select_node_best_split.inl>
#include <sah_kd_tree/separate_splitted_polygon.inl>
#include <sah_kd_tree/set_node_count.inl>
#include <sah_kd_tree/split_node.inl>
#include <sah_kd_tree/split_polygon.inl>
#include <sah_kd_tree/update_polygon_node.inl>
#include <sah_kd_tree/update_splitted_polygon_count.inl>
#include <sah_kd_tree/update_splitted_polygon_node.inl>
#endif

#pragma once

#include <builder/builder.hpp>
#include <compute/assert.hpp>
#include <compute/compute.hpp>
#include <sah_kd_tree/sah_kd_tree.cuh>
#include <scene_data/scene_data.hpp>
#include <utils/assert.hpp>
#include <utils/auto_cast.hpp>
#include <utils/demangle.hpp>
#include <utils/math.hpp>

#include <thrust/iterator/iterator_traits.h>
#include <thrust/iterator/strided_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/memory.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/tuple.h>
#include <thrust/uninitialized_copy.h>
#include <thrust/version.h>

#include <fmt/ranges.h>
#include <glm/fwd.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <functional>
#include <iterator>
#include <numeric>
#include <type_traits>
#include <typeinfo>
#include <utility>

#include <cstddef>

#include <cuda.h>

namespace builder
{

extern template TreePtr build<ThrustDeviceSystem::Default>(
    const Settings & settings,
    const compute::CudaDevice & cudaDevice,
    const scene_data::SceneDataPtr & sceneData,
    const std::function<bool(size_t progressValue)> & progress);
extern template TreePtr build<ThrustDeviceSystem::CPP>(
    const Settings & settings,
    const compute::CudaDevice & cudaDevice,
    const scene_data::SceneDataPtr & sceneData,
    const std::function<bool(size_t progressValue)> & progress);
extern template TreePtr build<ThrustDeviceSystem::OMP>(
    const Settings & settings,
    const compute::CudaDevice & cudaDevice,
    const scene_data::SceneDataPtr & sceneData,
    const std::function<bool(size_t progressValue)> & progress);
extern template TreePtr build<ThrustDeviceSystem::TBB>(
    const Settings & settings,
    const compute::CudaDevice & cudaDevice,
    const scene_data::SceneDataPtr & sceneData,
    const std::function<bool(size_t progressValue)> & progress);
extern template TreePtr build<ThrustDeviceSystem::CUDA>(
    const Settings & settings,
    const compute::CudaDevice & cudaDevice,
    const scene_data::SceneDataPtr & sceneData,
    const std::function<bool(size_t progressValue)> & progress);

template<typename BuilderContext>
struct GetTraits
{
    using Type = BuilderContext;
};

template<typename BuilderContext>
    requires requires { typename BuilderContext::Traits; }
struct GetTraits<BuilderContext>
{
    using Type = typename BuilderContext::Traits;
};

template<typename BuilderContext>
using GetTraitsType = typename GetTraits<BuilderContext>::Type;

template<ThrustDeviceSystem thrustDeviceSystem>
struct BuilderContext;

template<scene_data::Position::value_type scene_data::Position::* component>
struct VertexSlice
{
    __host__ __device__ auto operator()(const scene_data::Position & vertex)
    {
        return vertex.*component;
    }
};

template<
    typename F,
    typename Iterator>
void apply(
    F & f,
    Iterator it)
{
    f(it);
}

template<
    typename F,
    typename... Iterators>
void apply(
    F & f,
    thrust::zip_iterator<cuda::std::tuple<Iterators...>> zit)
{
    // TODO(tomilov): use cuda::std::apply after https://github.com/NVIDIA/cccl/issues/8038
    [&]<std::size_t... Is>(const cuda::std::tuple<Iterators...> & tuple, std::index_sequence<Is...>)
    {
        (builder::apply(f, cuda::std::get<Is>(tuple)), ...);
    }(zit.get_iterator_tuple(), std::index_sequence_for<Iterators...>{});
}

template<ThrustDeviceSystem thrustDeviceSystem>
struct Builder : Tree
{
    using BaseTraits = GetTraitsType<BuilderContext<thrustDeviceSystem>>;

    template<typename T>
    using Allocator = typename BaseTraits::template Allocator<T>;
    template<typename T>
    using Vector = typename BaseTraits::template Vector<T>;

    using System = typename thrust::iterator_system<typename Allocator<std::byte>::pointer>::type;
    static constexpr bool kIsThrustDeviceSystemCUDA = std::is_same_v<System, thrust::cuda::tag>;

    Builder(
        const Settings & settingsIn,
        const compute::CudaDevice & cudaDeviceIn,
        const scene_data::SceneDataPtr & sceneDataIn)
        : Tree{settingsIn,
              cudaDeviceIn,
              sceneDataIn}
    {
        ASSERT(sceneData);
    }

    [[nodiscard]] static auto getNode(const sah_kd_tree::Tree<BaseTraits> & tree)
    {
        auto aabbMin = thrust::make_zip_iterator(tree.x.node.min.data(), tree.y.node.min.data(), tree.z.node.min.data());
        auto aabbMax = thrust::make_zip_iterator(tree.x.node.max.data(), tree.y.node.max.data(), tree.z.node.max.data());
        auto leftRope = thrust::make_zip_iterator(tree.x.node.leftRope.data(), tree.y.node.leftRope.data(), tree.z.node.leftRope.data());
        auto rightRope = thrust::make_zip_iterator(tree.x.node.rightRope.data(), tree.y.node.rightRope.data(), tree.z.node.rightRope.data());
        auto splitDimension = tree.node.splitDimension.data();
        auto splitPos = tree.node.splitPos.data();
        auto leftChild = tree.node.leftChild.data();
        auto rightChild = tree.node.rightChild.data();
        return thrust::make_zip_iterator(aabbMin, aabbMax, leftRope, rightRope, splitDimension, splitPos, leftChild, rightChild);
    }

    template<typename T>
    size_t gatherSize(size_t count)
    {
        constexpr size_t kElementSize = sizeof(T);
        constexpr size_t kElementAlignment = kElementSize & (~kElementSize + 1);
        if (dataAlignment < kElementAlignment) {
            dataAlignment = kElementAlignment;
        }
        const size_t offset = utils::alignUp(dataSize, kElementAlignment);
        dataSize = offset + count * kElementSize;
        return offset;
    }

    template<typename T>
    size_t gatherSize(const Vector<T> & v)
    {
        return gatherSize<T>(std::size(v));
    }

    static void printThrustVersion()
    {
        SPDLOG_INFO("THRUST_VERSION: {}.{}.{}.{}", THRUST_MAJOR_VERSION, THRUST_MINOR_VERSION, THRUST_SUBMINOR_VERSION, THRUST_PATCH_NUMBER);
        {
            constexpr auto getHostSystemName = []() constexpr -> const char *
            {
                switch (THRUST_HOST_SYSTEM) {
                case THRUST_HOST_SYSTEM_OMP:
                    return "OMP";
                case THRUST_HOST_SYSTEM_TBB:
                    return "TBB";
                case THRUST_HOST_SYSTEM_CPP:
                    return "CPP";
                }
                return nullptr;
            };
            static_assert(getHostSystemName());
            SPDLOG_INFO("THRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_{}", getHostSystemName());
        }
        {
            constexpr auto getDeviceSystemName = []() constexpr -> const char *
            {
                switch (THRUST_DEVICE_SYSTEM) {
                case THRUST_DEVICE_SYSTEM_CUDA:
                    return "CUDA";
                case THRUST_DEVICE_SYSTEM_OMP:
                    return "OMP";
                case THRUST_DEVICE_SYSTEM_TBB:
                    return "TBB";
                case THRUST_DEVICE_SYSTEM_CPP:
                    return "CPP";
                }
                return nullptr;
            };
            static_assert(getDeviceSystemName());
            SPDLOG_INFO("THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_{}", getDeviceSystemName());
        }
    }

    template<
        typename T,
        size_t stride = 0>
    void gatherDeviceData(
        ::CUdeviceptr devPtr,
        const T * srcPtr,
        size_t count)
    {
        const ::CUdeviceptr srcDevPtr = utils::autoCast(srcPtr);
        if constexpr ((stride == sizeof(T) || (stride == 0))) {
            const size_t size = count * sizeof(T);
            if constexpr (kIsThrustDeviceSystemCUDA) {
                CU_CALL_CHECK(::cuMemcpyDtoD, devPtr, srcDevPtr, size);
            } else {
                CU_CALL_CHECK(::cuMemcpyHtoD, devPtr, srcPtr, size);
            }
        } else {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-designated-field-initializers"
            const ::CUDA_MEMCPY2D copyParams = {
                .srcMemoryType = kIsThrustDeviceSystemCUDA ? CU_MEMORYTYPE_DEVICE : CU_MEMORYTYPE_HOST,
                .srcHost = srcPtr,
                .srcDevice = srcDevPtr,
                .srcPitch = sizeof(T),
                .dstMemoryType = CU_MEMORYTYPE_DEVICE,
                .dstDevice = devPtr,
                .dstPitch = stride,
                .WidthInBytes = sizeof(T),
                .Height = count,
            };
#pragma GCC diagnostic pop
            ::cuMemcpy2D(&copyParams);
        }
    }

    template<typename... Types>
    void gatherDeviceData(
        ::CUdeviceptr devPtr,
        const Vector<Types> &... values)
    {
        size_t offsets[] = {0, sizeof(Types)...};
        std::inclusive_scan(std::cbegin(offsets), std::cend(offsets), std::begin(offsets));
        [&]<std::size_t... Is>(std::index_sequence<Is...>)
        {
            (gatherDeviceData<Types, (sizeof(Types) + ...)>(devPtr + offsets[Is], thrust::raw_pointer_cast(std::data(values)), std::size(values)), ...);
        }(std::index_sequence_for<Types...>{});
    }

    bool build(const std::function<bool(size_t progressValue)> & progress)
    {
        cudaDevice.setCurrentDevice();
        printThrustVersion();
        SPDLOG_INFO("BuilderContext: {}", utils::demangle(typeid(BuilderContext<thrustDeviceSystem>).name()));
        SPDLOG_INFO("system: {}", utils::demangle(typeid(System).name()));
        typename BuilderContext<thrustDeviceSystem>::TreeContext treeContext;
        {
            utils::MemArray<scene_data::Index> inputIndices;
            utils::MemArray<scene_data::Position> inputVertices;
            sceneData->collectScene(inputIndices, inputVertices);
            {
                const size_t indexCount = inputIndices.getCount();
                ASSERT((indexCount % 3) == 0);
                triangleCount = indexCount / 3;
                static_assert(std::is_same_v<typename scene_data::Index, typename BaseTraits::U>);
                auto a = inputIndices.cbegin();
                treeContext.index.a.resize(triangleCount);
                thrust::copy_n(thrust::make_strided_iterator<3>(a), triangleCount, treeContext.index.a.begin());
                auto b = cuda::std::next(a);
                treeContext.index.b.resize(triangleCount);
                thrust::copy_n(thrust::make_strided_iterator<3>(b), triangleCount, treeContext.index.b.begin());
                auto c = cuda::std::next(b);
                treeContext.index.c.resize(triangleCount);
                thrust::copy_n(thrust::make_strided_iterator<3>(c), triangleCount, treeContext.index.c.begin());
            }
            {
                vertexCount = inputVertices.getCount();
                static_assert(std::is_same_v<typename scene_data::Position::value_type, typename BaseTraits::F>);
                {
                    auto x = thrust::make_transform_iterator(inputVertices.cbegin(), VertexSlice<&scene_data::Position::x>{});
                    treeContext.vertex.x.resize(vertexCount);
                    thrust::copy_n(x, vertexCount, treeContext.vertex.x.begin());
                }
                {
                    auto y = thrust::make_transform_iterator(inputVertices.cbegin(), VertexSlice<&scene_data::Position::y>{});
                    treeContext.vertex.y.resize(vertexCount);
                    thrust::copy_n(y, vertexCount, treeContext.vertex.y.begin());
                }
                {
                    auto z = thrust::make_transform_iterator(inputVertices.cbegin(), VertexSlice<&scene_data::Position::z>{});
                    treeContext.vertex.z.resize(vertexCount);
                    thrust::copy_n(z, vertexCount, treeContext.vertex.z.begin());
                }
            }
        }
        {
            typename BuilderContext<thrustDeviceSystem>::BuildContext buildContext{treeContext};
            buildContext.builder.polygon.count = utils::autoCast(triangleCount);
            {
                buildContext.x.triangle.count = buildContext.builder.polygon.count;
                buildContext.x.triangle.a = thrust::make_permutation_iterator(treeContext.vertex.x.cbegin(), treeContext.index.a.cbegin());
                buildContext.x.triangle.b = thrust::make_permutation_iterator(treeContext.vertex.x.cbegin(), treeContext.index.b.cbegin());
                buildContext.x.triangle.c = thrust::make_permutation_iterator(treeContext.vertex.x.cbegin(), treeContext.index.c.cbegin());

                buildContext.y.triangle.count = buildContext.builder.polygon.count;
                buildContext.y.triangle.a = thrust::make_permutation_iterator(treeContext.vertex.y.cbegin(), treeContext.index.a.cbegin());
                buildContext.y.triangle.b = thrust::make_permutation_iterator(treeContext.vertex.y.cbegin(), treeContext.index.b.cbegin());
                buildContext.y.triangle.c = thrust::make_permutation_iterator(treeContext.vertex.y.cbegin(), treeContext.index.c.cbegin());

                buildContext.z.triangle.count = buildContext.builder.polygon.count;
                buildContext.z.triangle.a = thrust::make_permutation_iterator(treeContext.vertex.z.cbegin(), treeContext.index.a.cbegin());
                buildContext.z.triangle.b = thrust::make_permutation_iterator(treeContext.vertex.z.cbegin(), treeContext.index.b.cbegin());
                buildContext.z.triangle.c = thrust::make_permutation_iterator(treeContext.vertex.z.cbegin(), treeContext.index.c.cbegin());
            }
            const sah_kd_tree::Params<BaseTraits> params = {
                .emptinessFactor = settings.emptinessFactor,
                .traversalCost = settings.traversalCost,
                .intersectionCost = settings.intersectionCost,
                .maxTreeDepth = settings.maxTreeDepth,
            };
            if (!buildContext.builder.build(progress, params, buildContext.x, buildContext.y, buildContext.z, treeContext.tree)) {
                return false;
            }
        }
        static_assert(std::is_same_v<typename BaseTraits::F, glm::float32>);
        static_assert(std::is_same_v<typename BaseTraits::U, glm::uint32>);
        static_assert(std::is_same_v<typename BaseTraits::I, glm::int32>);
        const auto & tree = treeContext.tree;
        {
            ASSERT(std::is_sorted(std::cbegin(tree.layerDepth), std::cend(tree.layerDepth)));
            layerSizes.resize(std::size(tree.layerDepth));
            std::adjacent_difference(std::cbegin(tree.layerDepth), std::cend(tree.layerDepth), std::begin(layerSizes));
            polygonCount = std::size(tree.polygonTriangle);
            nodeCount = std::size(tree.node.parent);
            SPDLOG_INFO("Tree depth: {}", std::size(layerSizes));
            SPDLOG_INFO("Layer sizes: {}", layerSizes);
            SPDLOG_INFO("Triangle count: {}", triangleCount);
            SPDLOG_INFO("Vertex count: {}", vertexCount);
            SPDLOG_INFO("Polygon count: {}", polygonCount);
            SPDLOG_INFO("Node count: {}", nodeCount);
        }
        using Triangle = glm::uvec3;  // TODO: pass (future) scene_data::Triangle
        using Vertex = glm::vec3;     // TODO(tomilov): pass scene_data::Vertex
        indexOffset = gatherSize<Triangle>(triangleCount);
        vertexOffset = gatherSize<Vertex>(vertexCount);
        polygonOffset = gatherSize(tree.polygonTriangle);
        auto node = getNode(tree);
        using NodeType = cuda::std::iter_value_t<decltype(node)>;
        constexpr size_t kNodeSize = sizeof(NodeType);
        static_assert(kNodeSize == 64, "Keep in sync with Node in 'trace.comp'");
        const size_t maxPitch = cudaDevice.getMaxPitch();
        INVARIANT(kNodeSize < maxPitch, "{} ^ {}", kNodeSize, maxPitch);
        nodeOffset = gatherSize<NodeType>(nodeCount);
        nodeParentOffset = gatherSize(tree.node.parent);
        SPDLOG_INFO("Allocation size for tree: {}", dataSize);
        ASSERT(dataSize > 0);
        ASSERT(dataAlignment > 0);
        const auto deviceMemory = cudaDevice.makeDeviceMemory(dataSize, dataAlignment);
        allocationSize = deviceMemory.getSize();
        SPDLOG_INFO("Data size {}, data alignment {}, allocation size {}", dataSize, dataAlignment, allocationSize);
        {
            const auto mappedDeviceMemory = deviceMemory.map();
            const ::CUdeviceptr devPtr = mappedDeviceMemory.getCuDevPtr();
            gatherDeviceData(devPtr + indexOffset, treeContext.index.a, treeContext.index.b, treeContext.index.c);
            gatherDeviceData(devPtr + vertexOffset, treeContext.vertex.x, treeContext.vertex.y, treeContext.vertex.z);
            gatherDeviceData(devPtr + polygonOffset, tree.polygonTriangle);
            if constexpr (kIsThrustDeviceSystemCUDA) {
                const typename Allocator<NodeType>::pointer dst{utils::safeCast<NodeType *>(devPtr + nodeOffset)};
                thrust::uninitialized_copy_n(node, nodeCount, dst);
            } else {
                if ((false)) {
                    // Faster for CUDA, but CPP, OMP, TBB hangs (even if this code if never executed). TODO(tomilov): try later
                    auto dstPtr = devPtr;
                    const auto gatherNode = [this, &dstPtr]<typename Pointer>(Pointer srcPtr) -> void
                    {
                        gatherDeviceData(dstPtr, thrust::raw_pointer_cast(srcPtr), nodeCount);
                        dstPtr += sizeof(typename cuda::std::iterator_traits<Pointer>::value_type);
                    };
                    builder::apply(gatherNode, node);
                } else {
                    Vector<NodeType> nodes{tree.allocator};
                    nodes.assign(node, cuda::std::next(node, sah_kd_tree::safeConvert<ptrdiff_t>(nodeCount)));
                    auto srcPtr = thrust::raw_pointer_cast(nodes.data());
                    CU_CALL_CHECK(::cuMemcpyHtoD, devPtr + nodeOffset, srcPtr, nodes.size() * kNodeSize);
                }
            }
            gatherDeviceData(devPtr + nodeParentOffset, tree.node.parent);
            cudaDevice.synchronize();
        }
        deviceMemory.exportMemoryObject().swap(fd);
        return fd.has_value();
    }
};

template<ThrustDeviceSystem thrustDeviceSystem>
TreePtr build(
    const Settings & settings,
    const compute::CudaDevice & cudaDevice,
    const scene_data::SceneDataPtr & sceneData,
    const std::function<bool(size_t progressValue)> & progress)
{
    Builder<thrustDeviceSystem> Builder{settings, cudaDevice, sceneData};
    if (!Builder.build(progress)) {
        return nullptr;
    }
    return makeTreePtr(std::move(Builder));
}

}  // namespace builder

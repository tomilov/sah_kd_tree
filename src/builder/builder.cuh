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
#include <thrust/iterator/zip_iterator.h>
#include <thrust/memory.h>
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
        auto aabbMin = thrust::make_zip_iterator(tree.x.node.min.begin(), tree.y.node.min.begin(), tree.z.node.min.begin());
        auto aabbMax = thrust::make_zip_iterator(tree.x.node.max.begin(), tree.y.node.max.begin(), tree.z.node.max.begin());
        auto leftRope = thrust::make_zip_iterator(tree.x.node.leftRope.begin(), tree.y.node.leftRope.begin(), tree.z.node.leftRope.begin());
        auto rightRope = thrust::make_zip_iterator(tree.x.node.rightRope.begin(), tree.y.node.rightRope.begin(), tree.z.node.rightRope.begin());
        auto splitDimension = tree.node.splitDimension.begin();
        auto splitPos = tree.node.splitPos.begin();
        auto leftChild = tree.node.leftChild.begin();
        auto rightChild = tree.node.rightChild.begin();
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

    bool build(const std::function<bool(size_t progressValue)> & progress)
    {
        cudaDevice.setCurrentDevice();
        printThrustVersion();
        SPDLOG_INFO("BuilderContext: {}", utils::demangle(typeid(BuilderContext<thrustDeviceSystem>).name()));
        SPDLOG_INFO("system: {}", utils::demangle(typeid(System).name()));
        auto triangles = sceneData->makeTriangles();
        triangleCount = triangles.getCount();
        typename BuilderContext<thrustDeviceSystem>::TreeContext treeContext;
        {
            typename BuilderContext<thrustDeviceSystem>::BuildContext buildContext{treeContext};
            buildContext.triangle.setTriangle(triangles.begin(), triangles.end());
            sah_kd_tree::linkTriangles(buildContext.triangle, buildContext.x, buildContext.y, buildContext.z, buildContext.builder);
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
            SPDLOG_INFO("Polygon count: {}", polygonCount);
            SPDLOG_INFO("Node count: {}", nodeCount);
        }
        triangleOffset = gatherSize<scene_data::Triangle>(triangleCount);
        polygonOffset = gatherSize(tree.polygonTriangle);
        auto node = getNode(tree);
        using NodeType = cuda::std::iter_value_t<decltype(node)>;
        constexpr size_t kNodeSize = sizeof(NodeType);
        static_assert(kNodeSize == 64, "Keep in sync with Node in 'trace.comp'");
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
            const ::CUdeviceptr devPtr = mappedDeviceMemory.getPtr();
            const auto gatherDeviceData = [devPtr]<typename T>(size_t offset, const Vector<T> & v)
            {
                const T * const srcPtr = thrust::raw_pointer_cast(std::data(v));
                const size_t size = std::size(v) * sizeof(T);
                if constexpr (kIsThrustDeviceSystemCUDA) {
                    const ::CUdeviceptr src = utils::autoCast(srcPtr);
                    CU_CALL(::cuMemcpyDtoD, devPtr + offset, src, size);
                } else {
                    CU_CALL(::cuMemcpyHtoD, devPtr + offset, srcPtr, size);
                }
            };
            {
                constexpr size_t kTriangleSize = sizeof(scene_data::Triangle);
                CU_CALL(::cuMemcpyHtoD, devPtr + triangleOffset, triangles.begin(), triangleCount * kTriangleSize);
            }
            gatherDeviceData(polygonOffset, tree.polygonTriangle);
            {
                if constexpr (kIsThrustDeviceSystemCUDA) {
                    const typename Allocator<NodeType>::pointer dst{utils::safeCast<NodeType *>(devPtr + nodeOffset)};
                    thrust::uninitialized_copy_n(node, nodeCount, dst);
                } else {
                    Vector<NodeType> nodes{tree.allocator};
                    nodes.assign(node, cuda::std::next(node, sah_kd_tree::safeConvert<ptrdiff_t>(nodeCount)));
                    auto srcPtr = thrust::raw_pointer_cast(nodes.data());
                    CU_CALL(::cuMemcpyHtoD, devPtr + nodeOffset, srcPtr, nodes.size() * kNodeSize);
                }
            }
            gatherDeviceData(nodeParentOffset, tree.node.parent);
            CUDA_CALL(cudaDeviceSynchronize);
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

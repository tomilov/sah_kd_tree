#include <builder/builder.hpp>
#include <compute/compute.hpp>
#include <sah_kd_tree/sah_kd_tree.cuh>
#include <scene_data/scene_data.hpp>
#include <utils/assert.hpp>
#include <utils/auto_cast.hpp>
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
#include <utility>

#include <cstddef>

#include <cuda.h>

namespace builder
{

template<typename Traits>
struct GetTraits
{
    using Type = Traits;
};

template<typename Traits>
    requires requires { typename Traits::Traits; }
struct GetTraits<Traits>
{
    using Type = typename Traits::Traits;
};

template<typename Traits>
using GetTraitsType = typename GetTraits<Traits>::Type;

template<ThrustDeviceSystem Traits>
struct TreeBuildContext : Tree
{
    using SrcTraits = GetTraitsType<Traits>;

    static constexpr bool kIsThrustDeviceSystemCUDA = std::is_same_v<typename thrust::iterator_system<typename SrcTraits::template Allocator<std::byte>::pointer>::type, thrust::cuda::tag>;

    TreeBuildContext(const Settings & settings, const compute::CudaDevice & cudaDevice, const scene_data::SceneDataPtr & sceneData)
        : Tree{settings, cudaDevice, sceneData}
    {
        ASSERT(sceneData);
    }

    [[nodiscard]] static auto getNode(const sah_kd_tree::Tree<SrcTraits> & tree)
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
    size_t gatherSize(const typename SrcTraits::template Vector<T> & v)
    {
        return gatherSize<T>(std::size(v));
    }

    static void printThrustVersion()
    {
        SPDLOG_INFO("Thrust version: {}.{}.{}.{}", THRUST_MAJOR_VERSION, THRUST_MINOR_VERSION, THRUST_SUBMINOR_VERSION, THRUST_PATCH_NUMBER);
        {
            const char * hostSystem = nullptr;
            switch (THRUST_HOST_SYSTEM) {
            case THRUST_HOST_SYSTEM_OMP:
                hostSystem = "OMP";
                break;
            case THRUST_HOST_SYSTEM_TBB:
                hostSystem = "TBB";
                break;
            case THRUST_HOST_SYSTEM_CPP:
                hostSystem = "CPP";
                break;
            }
            INVARIANT(hostSystem, "{}", THRUST_HOST_SYSTEM);
            SPDLOG_INFO("Thrust host system: {}", hostSystem);
        }
        {
            const char * deviceSystem = nullptr;
            switch (THRUST_DEVICE_SYSTEM) {
            case THRUST_DEVICE_SYSTEM_CUDA:
                deviceSystem = "CUDA";
                break;
            case THRUST_DEVICE_SYSTEM_OMP:
                deviceSystem = "OMP";
                break;
            case THRUST_DEVICE_SYSTEM_TBB:
                deviceSystem = "TBB";
                break;
            case THRUST_DEVICE_SYSTEM_CPP:
                deviceSystem = "CPP";
                break;
            }
            INVARIANT(deviceSystem, "{}", THRUST_DEVICE_SYSTEM);
            SPDLOG_INFO("Thrust device system: {}", deviceSystem);
        }
    }

    bool build(const std::function<bool(size_t progressValue)> & progress)
    {
        cudaDevice.setCurrentDevice();
        printThrustVersion();
        SPDLOG_INFO("Builder kind: {}", __PRETTY_FUNCTION__);
        auto triangles = sceneData->makeTriangles();
        triangleCount = triangles.getCount();
        typename Traits::TreeContext treeContext;
        {
            typename Traits::BuildContext buildContext{treeContext};
            buildContext.triangle.setTriangle(triangles.begin(), triangles.end());
            sah_kd_tree::linkTriangles(buildContext.triangle, buildContext.x, buildContext.y, buildContext.z, buildContext.builder);
            const sah_kd_tree::Params<SrcTraits> params = {
                .emptinessFactor = settings.emptinessFactor,
                .traversalCost = settings.traversalCost,
                .intersectionCost = settings.intersectionCost,
                .maxTreeDepth = settings.maxTreeDepth,
            };
            if (!buildContext.builder.build(progress, params, buildContext.x, buildContext.y, buildContext.z, treeContext.tree)) {
                return false;
            }
        }
        static_assert(std::is_same_v<typename SrcTraits::F, glm::float32>);
        static_assert(std::is_same_v<typename SrcTraits::U, glm::uint32>);
        static_assert(std::is_same_v<typename SrcTraits::I, glm::int32>);
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
        const compute::DeviceMemory deviceMemory{cudaDevice.getCudaDriverDev(), dataSize, dataAlignment};
        allocationSize = deviceMemory.getSize();
        SPDLOG_INFO("Data size {}, data alignment {}, allocation size {}", dataSize, dataAlignment, allocationSize);
        {
            const auto mappedDeviceMemory = deviceMemory.map();
            const ::CUdeviceptr devPtr = mappedDeviceMemory.getPtr();
            const auto gatherDeviceData = [devPtr]<typename T>(size_t offset, const typename SrcTraits::template Vector<T> & v)
            {
                const T * const srcPtr = thrust::raw_pointer_cast(std::data(v));
                const size_t size = std::size(v) * sizeof(T);
                if constexpr (kIsThrustDeviceSystemCUDA) {
                    const ::CUdeviceptr src = utils::autoCast(srcPtr);
                    CU_CHECK_ERROR(::cuMemcpyDtoD, devPtr + offset, src, size);
                } else {
                    CU_CHECK_ERROR(::cuMemcpyHtoD, devPtr + offset, srcPtr, size);
                }
            };
            {
                constexpr size_t kTriangleSize = sizeof(scene_data::Triangle);
                CU_CHECK_ERROR(::cuMemcpyHtoD, devPtr + triangleOffset, triangles.begin(), triangleCount * kTriangleSize);
            }
            gatherDeviceData(polygonOffset, tree.polygonTriangle);
            {
                if constexpr (kIsThrustDeviceSystemCUDA) {
                    const typename SrcTraits::template Allocator<NodeType>::pointer dst{utils::safeCast<NodeType *>(devPtr + nodeOffset)};
                    thrust::uninitialized_copy_n(node, nodeCount, dst);
                } else {
                    typename SrcTraits::template Vector<NodeType> nodes{tree.allocator};
                    nodes.assign(node, cuda::std::next(node, nodeCount));
                    auto srcPtr = thrust::raw_pointer_cast(nodes.data());
                    CU_CHECK_ERROR(::cuMemcpyHtoD, devPtr + nodeOffset, srcPtr, nodes.size() * kNodeSize);
                }
            }
            gatherDeviceData(nodeParentOffset, tree.node.parent);
            CUDA_CHECK_ERROR(cudaDeviceSynchronize);
        }
        fd.emplace(deviceMemory.exportMemoryObject());
        return true;
    }
};

template<ThrustDeviceSystem Traits>
TreePtr build(const Settings & settings, const compute::CudaDevice & cudaDevice, const scene_data::SceneDataPtr & sceneData, const std::function<bool(size_t progressValue)> & progress)
{
    TreeBuildContext<Traits> treeBuildContext{settings, cudaDevice, sceneData};
    if (!treeBuildContext.build(progress)) {
        return nullptr;
    }
    return makeTreePtr(std::move(treeBuildContext));
}

}  // namespace builder

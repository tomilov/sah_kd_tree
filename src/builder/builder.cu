#include <builder/builder.hpp>
#include <compute/compute.hpp>
#if SAH_KD_TREE_BUILDER_USE_DEFAULT_TRAITS
#include <sah_kd_tree/sah_kd_tree.cuh>
#else
#include <sah_kd_tree/sah_kd_tree_inline.cuh>
#endif
#include <scene_data/scene_data.hpp>
#include <utils/assert.hpp>
#include <utils/auto_cast.hpp>
#include <utils/math.hpp>

#include <thrust/iterator/zip_iterator.h>
#if !SAH_KD_TREE_BUILDER_USE_DEFAULT_TRAITS
#include <thrust/device_vector.h>
#include <thrust/mr/allocator.h>
#include <thrust/mr/device_memory_resource.h>
#endif
#include <thrust/memory.h>
#include <thrust/uninitialized_copy.h>

#include <fmt/ranges.h>
#include <glm/fwd.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <bit>
#include <functional>
#include <iterator>
#include <memory>
#include <numeric>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <cstddef>

#include <cuda.h>
#include <cuda_runtime.h>

namespace builder
{
namespace
{

#if SAH_KD_TREE_BUILDER_USE_DEFAULT_TRAITS
using Traits = sah_kd_tree::DefaultTraits;
#else
struct Traits  // cannot be member typedef of Tree::Impl because of wierd CUDA parser
{
    using F = sah_kd_tree::DefaultTraits::F;
    using I = sah_kd_tree::DefaultTraits::I;
    using U = sah_kd_tree::DefaultTraits::U;
    using MemoryResource = thrust::device_memory_resource;
    template<typename T>
    using Allocator = thrust::mr::allocator<T, MemoryResource>;
    template<typename T>
    using Vector = thrust::device_vector<T, Allocator<T>>;
    using Progress = sah_kd_tree::DefaultTraits::Progress;
};
#endif

}  // namespace

struct Tree::Impl : utils::OneTime<Impl>
{
    const Settings settings;
    const compute::CudaDevice & cudaDevice;
    const scene_data::SceneDataWeakPtr sceneData;

    size_t dataSize = 0;
    size_t dataAlignment = 0;
    size_t allocationSize = 0;

    size_t triangleCount = 0;
    std::vector<size_t> layerSizes;
    size_t polygonCount = 0;
    size_t nodeCount = 0;

    size_t triangleOffset = 0;
    size_t polygonOffset = 0;
    size_t nodeOffset = 0;
    size_t nodeParentOffset = 0;

    std::optional<utils::Fd> fd;

    [[nodiscard]] bool operator==(const Impl & rhs) const noexcept
    {
        return std::forward_as_tuple(settings, cudaDevice, sceneData.lock()) == std::forward_as_tuple(rhs.settings, rhs.cudaDevice, rhs.sceneData.lock());
    }

    [[nodiscard]] static auto getNode(const sah_kd_tree::Tree<Traits> & tree)
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
        constexpr size_t kElementAlignment = static_cast<size_t>(1) << std::countr_zero(kElementSize);
        if (dataAlignment < kElementAlignment) {
            dataAlignment = kElementAlignment;
        }
        const size_t offset = utils::divUp(dataSize, kElementAlignment) * kElementAlignment;
        dataSize = offset + count * kElementSize;
        return offset;
    }

    template<typename T>
    size_t gatherSize(const typename Traits::template Vector<T> & v)
    {
        return gatherSize<T>(std::size(v));
    }

    static void printThrustVersion()
    {
        [[maybe_unused]] int major = THRUST_MAJOR_VERSION;
        [[maybe_unused]] int minor = THRUST_MINOR_VERSION;
        [[maybe_unused]] int subminor = THRUST_SUBMINOR_VERSION;
        [[maybe_unused]] int patch = THRUST_PATCH_NUMBER;
        SPDLOG_DEBUG("Thrust version: {}.{}.{}.{}", major, minor, subminor, patch);
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
        SPDLOG_DEBUG("Thrust device system: {}", deviceSystem);
    }

    Impl(const Settings & settings, const compute::CudaDevice & cudaDevice, const scene_data::SceneDataPtr & sceneData, const std::function<bool(size_t progressValue)> & progress)
        : settings{settings}
        , cudaDevice{cudaDevice}
        , sceneData{sceneData}
    {
        CUDA_CHECK_ERROR(cudaSetDevice(cudaDevice.getCudaRuntimeDev()));
        printThrustVersion();
        ASSERT(sceneData);
        auto triangles = sceneData->makeTriangles();
        triangleCount = triangles.getCount();
#if SAH_KD_TREE_BUILDER_USE_DEFAULT_TRAITS
        sah_kd_tree::Tree<Traits> tree;
#else
        typename Traits::MemoryResource memoryResource;
        typename Traits::Allocator<void> allocator{&memoryResource};

        sah_kd_tree::Tree<Traits> tree{allocator};
#endif
        {
#if SAH_KD_TREE_BUILDER_USE_DEFAULT_TRAITS
            sah_kd_tree::Builder<Traits> builder;
            sah_kd_tree::Projection<Traits> x, y, z;

            sah_kd_tree::Triangle<Traits> triangle;
#else
            sah_kd_tree::Builder<Traits> builder{allocator};
            sah_kd_tree::Projection<Traits> x{allocator}, y{allocator}, z{allocator};

            sah_kd_tree::Triangle<Traits> triangle{allocator};
#endif
            triangle.setTriangle(triangles.begin(), triangles.end());
            sah_kd_tree::linkTriangles(triangle, x, y, z, builder);
            const sah_kd_tree::Params<Traits> params = {
                .emptinessFactor = settings.emptinessFactor,
                .traversalCost = settings.traversalCost,
                .intersectionCost = settings.intersectionCost,
                .maxTreeDepth = settings.maxTreeDepth,
            };
            if (!builder.build(progress, params, x, y, z, tree)) {
                return;
            }
            // free device memory occupied by temporary arrays required for tree building
        }
        static_assert(std::is_same_v<Traits::F, glm::float32>);
        static_assert(std::is_same_v<Traits::U, glm::uint32>);
        static_assert(std::is_same_v<Traits::I, glm::int32>);
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
            const auto gatherDeviceData = [devPtr]<typename T>(size_t offset, const typename Traits::template Vector<T> & v)
            {
                const T * const srcPtr = thrust::raw_pointer_cast(std::data(v));
                const size_t size = std::size(v) * sizeof(T);
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
                const ::CUdeviceptr src = utils::autoCast(srcPtr);
                CU_CHECK_ERROR(::cuMemcpyDtoD(devPtr + offset, src, size));
#else
                CU_CHECK_ERROR(::cuMemcpyHtoD(devPtr + offset, srcPtr, size));
#endif
            };
            {
                constexpr size_t kTriangleSize = sizeof(scene_data::Triangle);
                CU_CHECK_ERROR(::cuMemcpyHtoD(devPtr + triangleOffset, triangles.begin(), triangleCount * kTriangleSize));
            }
            gatherDeviceData(polygonOffset, tree.polygonTriangle);
            {
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
                const Traits::Allocator<NodeType>::pointer dst{utils::safeCast<NodeType *>(devPtr + nodeOffset)};
                thrust::uninitialized_copy_n(node, nodeCount, dst);
#else
                typename Traits::Vector<NodeType> nodes{tree.allocator};
                nodes.assign(node, cuda::std::next(node, nodeCount));
                auto srcPtr = thrust::raw_pointer_cast(nodes.data());
                CU_CHECK_ERROR(::cuMemcpyHtoD(devPtr + nodeOffset, srcPtr, nodes.size() * kNodeSize));
#endif
            }
            gatherDeviceData(nodeParentOffset, tree.node.parent);
            CUDA_CHECK_ERROR(cudaDeviceSynchronize());
        }
        fd.emplace(deviceMemory.exportMemoryObject());
    }

    Impl(Impl &&) noexcept = default;

    static constexpr void completeClassContext [[maybe_unused]] ()
    {
        checkTraits();
    }
};

Tree::Tree(const Settings & settings, const compute::CudaDevice & cudaDevice, const scene_data::SceneDataPtr & sceneData, const std::function<bool(size_t progressValue)> & progress)
    : impl_{std::make_unique<Impl>(settings, cudaDevice, sceneData, progress)}
{}

Tree::Tree(Tree &&) noexcept = default;

Tree::~Tree() = default;

auto Tree::getSettings() const & -> const Settings &
{
    return impl_->settings;
}

const compute::CudaDevice & Tree::getCudaDevice() const &
{
    return impl_->cudaDevice;
}

scene_data::SceneDataPtr Tree::getSceneData() const
{
    return impl_->sceneData.lock();
}

bool Tree::operator==(const Tree & rhs) const noexcept
{
    return *impl_ == *rhs.impl_;
}

size_t Tree::getTriangleCount() const
{
    ASSERT(impl_->triangleCount > 0);
    return impl_->triangleCount;
}

const std::vector<size_t> & Tree::getLayerSizes() const &
{
    ASSERT(!std::empty(impl_->layerSizes));
    return impl_->layerSizes;
}

size_t Tree::getPolygonCount() const
{
    ASSERT(impl_->polygonCount > 0);
    return impl_->polygonCount;
}

size_t Tree::getNodeCount() const
{
    ASSERT(impl_->nodeCount > 0);
    return impl_->nodeCount;
}

size_t Tree::getDataSize() const
{
    ASSERT(impl_->dataSize > 0);
    return impl_->dataSize;
}

size_t Tree::getDataAlignment() const
{
    ASSERT(impl_->dataAlignment > 0);
    return impl_->dataAlignment;
}

size_t Tree::getAllocationSize() const
{
    ASSERT(impl_->allocationSize > 0);
    return impl_->allocationSize;
}

size_t Tree::getTriangleOffset() const
{
    return impl_->triangleOffset;
}

size_t Tree::getPolygonOffset() const
{
    return impl_->polygonOffset;
}

size_t Tree::getNodeOffset() const
{
    return impl_->nodeOffset;
}

size_t Tree::getNodeParentOffset() const
{
    return impl_->nodeParentOffset;
}

bool Tree::isEmpty() const
{
    return !impl_->fd;
}

utils::Fd Tree::stealFd() &&
{
    ASSERT(!isEmpty());
    utils::Fd fd = std::move(impl_->fd).value();
    impl_->fd.reset();
    return fd;
}

utils::Fd Tree::cloneFd() const
{
    ASSERT(!isEmpty());
    return impl_->fd.value().clone();
}

void TreeDeleter::operator()(Tree * tree) const noexcept
{
    return std::default_delete<Tree>{}(tree);
}

TreePtr makeTreePtr(Tree && tree)
{
    return {new Tree{std::move(tree)}, TreeDeleter{}};
}

}  // namespace builder

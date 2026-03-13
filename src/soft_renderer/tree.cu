#include <builder/builder.hpp>
#include <compute/assert.hpp>
#include <compute/compute.hpp>
#include <scene_data/scene_data.hpp>
#include <soft_renderer/tree.hpp>
#include <utils/auto_cast.hpp>

#include <utility>

#include <cstddef>
#include <cstring>

#include <cuda.h>
#include <cuda_runtime.h>

namespace soft_renderer
{

void importTree(
    builder::Tree tree,
    utils::MemArray<scene_data::Triangle> & triangles,
    utils::MemArray<glm::uint> & polygons,
    utils::MemArray<Node> & nodes,
    utils::MemArray<glm::uint> & nodeParents)
{
    const auto deviceMemory = tree.cudaDevice.makeDeviceMemory(std::move(tree).fd.value(), tree.allocationSize, tree.dataAlignment);
    const auto mappedDeviceMemory = deviceMemory.map();
    const ::CUdeviceptr devPtr = mappedDeviceMemory.getCuDevPtr();
    const auto scatterDeviceData = [devPtr]<typename T>(size_t offset, size_t count, utils::MemArray<T> & v)
    {
        v = utils::MemArray<T>{count};
        CU_CALL_CHECK(::cuMemcpyDtoH, v.begin(), devPtr + offset, count * sizeof(T));
    };
    scatterDeviceData(tree.triangleOffset, tree.triangleCount, triangles);
    scatterDeviceData(tree.polygonOffset, tree.polygonCount, polygons);
    scatterDeviceData(tree.nodeOffset, tree.nodeCount, nodes);
    scatterDeviceData(tree.nodeParentOffset, tree.nodeCount, nodeParents);
    compute::CudaDevice::synchronize();
}

}  // namespace soft_renderer

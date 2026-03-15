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
    utils::MemArray<glm::uvec3> & indices,
    utils::MemArray<glm::vec3> & vertices,
    utils::MemArray<glm::uint> & polygons,
    utils::MemArray<Node> & nodes,
    utils::MemArray<glm::uint> & nodeParents)
{
    const auto cudaStream = compute::CudaStream::make();
    const auto deviceMemory = tree.cudaDevice.makeDeviceMemory(std::move(tree).fd.value(), tree.allocationSize, tree.dataAlignment);
    const auto mappedDeviceMemory = deviceMemory.map();
    const ::CUdeviceptr devPtr = mappedDeviceMemory.getCuDevPtr();
    const auto scatterDeviceData = [&cudaStream, devPtr]<typename T>(size_t offset, size_t count, utils::MemArray<T> & v)
    {
        v.setCount(count);
        CU_CALL_CHECK(::cuMemcpyDtoHAsync, v.begin(), devPtr + offset, count * sizeof(T), cudaStream.getHandle());
    };
    scatterDeviceData(tree.indexOffset, tree.triangleCount, indices);
    scatterDeviceData(tree.vertexOffset, tree.vertexCount, vertices);
    scatterDeviceData(tree.polygonOffset, tree.polygonCount, polygons);
    scatterDeviceData(tree.nodeOffset, tree.nodeCount, nodes);
    scatterDeviceData(tree.nodeParentOffset, tree.nodeCount, nodeParents);
    cudaStream.synchronize();
}

}  // namespace soft_renderer

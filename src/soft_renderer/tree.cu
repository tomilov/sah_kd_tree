#include <builder/builder.hpp>
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

void importTree(builder::Tree tree, utils::MemArray<scene_data::Triangle> & triangles, utils::MemArray<glm::uint> & polygons, utils::MemArray<Node> & nodes, utils::MemArray<glm::uint> & nodeParents)
{
    const compute::CudaDevice & cudaDevice = tree.getCudaDevice();
    compute::DeviceMemory deviceMemory{cudaDevice.getCudaDriverDev(), std::move(tree).stealFd(), tree.getAllocationSize(), tree.getDataAlignment()};
    const auto mappedDeviceMemory = deviceMemory.map();
    const ::CUdeviceptr devPtr = mappedDeviceMemory.getPtr();
    const auto scatterDeviceData = [devPtr]<typename T>(size_t offset, size_t count, utils::MemArray<T> & v)
    {
        v = utils::MemArray<T>{count};
        CU_CHECK_ERROR(::cuMemcpyDtoH, v.begin(), devPtr + offset, count * sizeof(T));
    };
    scatterDeviceData(tree.getTriangleOffset(), tree.getTriangleCount(), triangles);
    scatterDeviceData(tree.getPolygonOffset(), tree.getPolygonCount(), polygons);
    scatterDeviceData(tree.getNodeOffset(), tree.getNodeCount(), nodes);
    scatterDeviceData(tree.getNodeParentOffset(), tree.getNodeCount(), nodeParents);
    CUDA_CHECK_ERROR(cudaDeviceSynchronize);
}

}  // namespace soft_renderer

#include <builder/builder.hpp>
#include <compute/compute.hpp>
#include <scene_data/scene_data.hpp>
#include <soft_renderer/tree.hpp>
#include <utils/auto_cast.hpp>

#include <iterator>
#include <utility>
#include <vector>

#include <cstddef>
#include <cstring>

#include <cuda.h>
#include <cuda_runtime.h>

namespace soft_renderer
{

void importTree(builder::Tree tree, std::vector<scene_data::Triangle> & triangles, std::vector<glm::uint> & polygons, std::vector<Node> & nodes, std::vector<glm::uint> & nodeParents)
{
    const compute::CudaDevice & cudaDevice = tree.getCudaDevice();
    compute::DeviceMemory deviceMemory{cudaDevice.getCudaDriverDev(), std::move(tree).stealFd(), tree.getAllocationSize(), tree.getDataAlignment()};
    const auto mappedDeviceMemory = deviceMemory.map();
    const ::CUdeviceptr devPtr = mappedDeviceMemory.getPtr();
    const auto scatterDeviceData = [devPtr]<typename T>(size_t offset, size_t count, std::vector<T> & v)
    {
        v.resize(count);
        CU_CHECK_ERROR(::cuMemcpyDtoH(std::data(v), devPtr + offset, count * sizeof(T)));
    };
    scatterDeviceData(tree.getTriangleOffset(), tree.getTriangleCount(), triangles);
    scatterDeviceData(tree.getPolygonOffset(), tree.getPolygonCount(), polygons);
    scatterDeviceData(tree.getNodeOffset(), tree.getNodeCount(), nodes);
    scatterDeviceData(tree.getNodeParentOffset(), tree.getNodeCount(), nodeParents);
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
}

}  // namespace soft_renderer

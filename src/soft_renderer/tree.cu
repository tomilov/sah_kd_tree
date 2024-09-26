#include <builder/builder.hpp>
#include <compute/compute.hpp>
#include <scene_data/scene_data.hpp>
#include <soft_renderer/tree.hpp>
#include <utils/auto_cast.hpp>

#include <cstddef>
#include <cstring>

#include <cuda.h>
#include <cuda_runtime.h>

namespace soft_renderer
{

void importTree(const builder::Tree & tree, std::vector<scene_data::Triangle> & triangles, std::vector<glm::uint> & polygons, std::vector<Node> & nodes, std::vector<glm::uint> & nodeParents)
{
    triangles.resize(tree.getTriangleCount());
    // std::vector<size_t> layerSizes = tree.getLayerSizes();
    polygons.resize(tree.getPolygonCount());
    nodes.resize(tree.getNodeCount());
    nodeParents.resize(tree.getNodeCount());

    const compute::CudaDevice & cudaDevice = tree.getCudaDevice();
    compute::DeviceMemory deviceMemory{cudaDevice.getCudaDriverDev(), tree.cloneFd(), tree.getAllocationSize(), tree.getDataAlignment()};

    // const vk::DeviceSize dataSize = utils::autoCast(tree.getDataSize());

    const auto mappedDeviceMemory = deviceMemory.map();
    const ::CUdeviceptr devPtr = mappedDeviceMemory.getPtr();

    const auto scatterDeviceData = [devPtr]<typename T>(size_t offset, std::vector<T> & v)
    {
        CU_CHECK_ERROR(::cuMemcpyDtoH(std::data(v), devPtr + offset, std::size(v) * sizeof(T)));
    };
    scatterDeviceData(tree.getTriangleOffset(), triangles);
    scatterDeviceData(tree.getPolygonOffset(), polygons);
    scatterDeviceData(tree.getNodeOffset(), nodes);
    scatterDeviceData(tree.getNodeParentOffset(), nodeParents);

    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
}

}  // namespace soft_renderer

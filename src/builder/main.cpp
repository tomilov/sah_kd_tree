#include <builder/builder.hpp>
#include <compute/compute.hpp>
#include <compute/make.hpp>
#include <scene_data/scene_data.hpp>

#include <cstddef>

int main()
{
    compute::CudaDevicePtr cudaDevice = compute::makeCudaDevice(std::nullopt);
    builder::Tree::Settings settings = {

    };
    scene_data::SceneDataPtr sceneData = nullptr;
    const auto progress = []([[maybe_unused]] size_t progressValue) -> bool
    {
        return false;
    };
    builder::Tree tree{settings, *cudaDevice, sceneData, progress};
}

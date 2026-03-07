#include <builder/builder.hpp>
#include <compute/compute.hpp>
#include <compute/make.hpp>
#include <scene_data/scene_data.hpp>
#include <utils/assert.hpp>

#include <cstddef>

int main()
{
    compute::CudaDevicePtr cudaDevice = compute::makeCudaDevice(std::nullopt);
    builder::Settings settings = {

    };
    scene_data::SceneDataPtr sceneData = nullptr;
    const auto progress = []([[maybe_unused]] size_t progressValue) -> bool
    {
        return false;
    };
    for (size_t i = 0; i < 5; ++i) {
        auto build = builder::getBuild(i);
        INVARIANT(build, "{}", i);
        auto tree = build(settings, *cudaDevice, sceneData, progress);
    }
}

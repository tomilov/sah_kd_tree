#include <renderer/context.hpp>
#include <renderer/memory.hpp>
#include <renderer/renderer.hpp>

#include <common/version.hpp>

#include <vulkan/vulkan.hpp>

int main(int /*argc*/, char * /*argv*/[])
{
    renderer::Renderer renderer;

    renderer::Context context;
    constexpr auto kApplicationVersion = VK_MAKE_VERSION(sah_kd_tree::kProjectVersionMajor, sah_kd_tree::kProjectVersionMinor, sah_kd_tree::kProjectVersionPatch);
    renderer::AllocationCallbacks allocationCallbacks;
    {
        using A = renderer::Allocator<int, vk::SystemAllocationScope::eInstance>;
        A a{allocationCallbacks.allocationCallbacks};
        std::vector<int, A> v{a};
        v.push_back(1);
    }
    context.createInstance(APPLICATION_NAME, kApplicationVersion, allocationCallbacks.allocationCallbacks);
    context.createDevice();
}

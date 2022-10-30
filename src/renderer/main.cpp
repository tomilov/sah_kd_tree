#include <renderer/context.hpp>
#include <renderer/renderer.hpp>

#include <common/version.hpp>

#include <vulkan/vulkan.hpp>

int main(int /*argc*/, char * /*argv*/[])
{
    renderer::Renderer renderer;

    renderer::Context context;
    context.init(APPLICATION_NAME, VK_MAKE_VERSION(sah_kd_tree::kProjectVersionMajor, sah_kd_tree::kProjectVersionMinor, sah_kd_tree::kProjectVersionPatch));
}

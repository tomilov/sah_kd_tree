#include <soft_renderer/soft_renderer.hpp>

#include <glm/vec4.hpp>

#include <string_view>

using namespace std::string_view_literals;

int main()
{
    const glm::vec4 kClearColor{0.0f, 0.0f, 0.0f, 1.0f};
    soft_renderer::SoftRenderer softRenderer{"default"sv, kClearColor};
}

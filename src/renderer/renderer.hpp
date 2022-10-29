#pragma once

#include <renderer/renderer_export.h>
#include <utils/fast_pimpl.hpp>

#include <vulkan/vulkan.hpp>

namespace renderer
{
class RENDERER_EXPORT Renderer
{
public:
    Renderer();
    ~Renderer();

private:
    struct Impl;

    static constexpr std::size_t kSize = 1;
    static constexpr std::size_t kAlignment = 1;
    utils::FastPimpl<Impl, 1, 1> impl_;
};

}  // namespace renderer

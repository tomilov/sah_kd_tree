#pragma once

#include <memory>

namespace softrenderer
{
class SoftRenderer;

using SoftRendererPtr = std::shared_ptr<const SoftRenderer>;
using SoftRendererWeakPtr = std::weak_ptr<const SoftRenderer>;
}  // namespace softrenderer

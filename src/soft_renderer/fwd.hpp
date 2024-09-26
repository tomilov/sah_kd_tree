#pragma once

#include <memory>

namespace soft_renderer
{
class SoftRenderer;

using SoftRendererPtr = std::shared_ptr<const SoftRenderer>;
using SoftRendererWeakPtr = std::weak_ptr<const SoftRenderer>;
}  // namespace soft_renderer

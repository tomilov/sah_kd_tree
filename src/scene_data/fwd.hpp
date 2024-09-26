#pragma once

#include <memory>

namespace scene_data
{
struct Triangle;
struct SceneData;

using SceneDataPtr = std::shared_ptr<const SceneData>;
using SceneDataWeakPtr = std::weak_ptr<const SceneData>;
}  // namespace scene_data

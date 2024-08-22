#pragma once

#include <scene_data/fwd.hpp>
#include <utils/noncopyable.hpp>

#include <filesystem>
#include <mutex>
#include <unordered_map>

namespace viewer
{

class Scenes : utils::NonCopyable
{
public:
    [[nodiscard]] scene_data::SceneDataPtr getScene(const std::filesystem::path & scenePath) const;

private:
    mutable std::mutex mutex;
    mutable std::unordered_map<std::filesystem::path, std::weak_ptr<scene_data::SceneData>> scenes;
};

}  // namespace viewer

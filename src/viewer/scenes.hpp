#pragma once

#include <scene_data/scene_data.hpp>
#include <utils/noncopyable.hpp>

#include <filesystem>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace viewer
{

struct Scene : utils::OneTime<Scene>
{
    std::filesystem::path scenePath;
    scene_data::SceneData sceneData;

    Scene(std::filesystem::path scenePath, scene_data::SceneData && sceneData)
        : scenePath{std::move(scenePath)}
        , sceneData{std::move(sceneData)}
    {}

    static constexpr void completeClassContext()
    {
        checkTraits();
    }
};

class Scenes : utils::NonCopyable
{
public:
    [[nodiscard]] std::shared_ptr<const Scene> getScene(const std::filesystem::path & scenePath) const;

private:
    mutable std::mutex mutex;
    mutable std::unordered_map<std::filesystem::path, std::weak_ptr<const Scene>> scenes;
};

}  // namespace viewer

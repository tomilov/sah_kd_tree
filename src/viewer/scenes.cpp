#include <scene_loader/scene_loader.hpp>
#include <utils/assert.hpp>
#include <viewer/scenes.hpp>

#include <fmt/std.h>
#include <spdlog/spdlog.h>

#include <QFileInfo>
#include <QStandardPaths>

#include <iterator>
#include <memory>
#include <mutex>
#include <utility>

using namespace std::string_literals;
using namespace std::string_view_literals;

namespace viewer
{

std::shared_ptr<Scene> Scenes::getScene(const std::filesystem::path & scenePath) const
{
    ASSERT(!std::empty(scenePath));
    std::lock_guard<std::mutex> lockGuard{mutex};
    auto & w = scenes[scenePath];
    auto p = w.lock();
    if (p) {
        SPDLOG_TRACE("Old scene {} reused", scenePath);
    } else {
        Scene scene;
        if ((true)) {
            auto cacheLocation = QStandardPaths::writableLocation(QStandardPaths::CacheLocation);
            if (!scene_loader::cachingLoad(scene.sceneData, QFileInfo{scenePath}, cacheLocation)) {
                return nullptr;
            }
        } else {
            if (!scene_loader::load(scene.sceneData, QFileInfo{scenePath})) {
                return nullptr;
            }
        }
        p = std::make_shared<Scene>(std::move(scene));
        w = p;
        SPDLOG_TRACE("New scene {} created", scenePath);
    }
    return p;
}

}  // namespace viewer

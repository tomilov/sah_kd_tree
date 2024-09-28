#include <builder/builder.hpp>
#include <compute/compute.hpp>
#include <compute/make.hpp>
#include <scene_data/scene_data.hpp>
#include <scene_loader/scene_loader.hpp>
#include <soft_renderer/soft_renderer.hpp>
#include <utils/assert.hpp>
#include <utils/auto_cast.hpp>

#include <gli/texture2d.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include <QtCore/QDebug>
#include <QtCore/QDir>
#include <QtCore/QFileInfo>
#include <QtCore/QLoggingCategory>
#include <QtCore/QStandardPaths>
#include <QtCore/QString>
#include <QtCore/QtLogging>

#include <string_view>

#include <cstdlib>

using namespace Qt::StringLiterals;
using namespace std::string_view_literals;

namespace
{
Q_DECLARE_LOGGING_CATEGORY(softRendererMain)
Q_LOGGING_CATEGORY(softRendererMain, "soft_renderer.main")

constexpr float kEmptinessFactor = 0.8f;
constexpr float kTraversalCost = 2.0f;
constexpr float kIntersectionCost = 1.0f;
constexpr uint32_t kMaxdepth = 100;

builder::TreePtr makeTree(QString sceneFileName)
{
    compute::CudaDevicePtr cudaDevice = compute::makeCudaDevice(std::nullopt);
    scene_data::SceneData sceneData;
    QFileInfo sceneFileInfo{sceneFileName};
    if ((true)) {
        if (!scene_loader::load(sceneData, sceneFileInfo)) {
            qCDebug(softRendererMain).noquote() << u"Cannot load scene from file %1"_s.arg(sceneFileName);
            return nullptr;
        }
    } else {
        auto cachePath = QStandardPaths::writableLocation(QStandardPaths::CacheLocation);
        if (!scene_loader::cachingLoad(sceneData, sceneFileInfo, cachePath.isEmpty() ? QDir::temp() : cachePath)) {
            qCDebug(softRendererMain).noquote() << u"Cannot load scene from file %1"_s.arg(sceneFileName);
            return nullptr;
        }
    }
    builder::Tree::Settings settings = {
        .emptinessFactor = kEmptinessFactor,
        .traversalCost = kTraversalCost,
        .intersectionCost = kIntersectionCost,
        .maxDepth = kMaxdepth,
    };
    const auto progress = [start = std::chrono::steady_clock::now()](size_t progressValue)
    {
        using namespace std::chrono_literals;
        if (start + 10s < std::chrono::steady_clock::now()) {
            INVARIANT(false, "{}", progressValue);
        }
        return false;
    };
    builder::Tree tree{settings, *cudaDevice, std::make_shared<scene_data::SceneData>(std::move(sceneData)), progress};
    if (tree.isEmpty()) {
        return nullptr;
    }
    return std::make_shared<builder::Tree>(std::move(tree));
}

}  // namespace

int main(int argc, char * argv[])
{
    INVARIANT(argc == 2, "{}", argc);
    builder::TreePtr tree = makeTree(QString::fromUtf8(argv[1]));
    if (!tree) {
        return EXIT_FAILURE;
    }
    const glm::vec4 kClearColor{0.0f, 0.0f, 0.0f, 1.0f};
    soft_renderer::SoftRenderer softRenderer{"default"sv, kClearColor};
    softRenderer.updateTree(tree);
    soft_renderer::FrameSettings frameSettings = {
        .position = glm::vec3{0.0f, 0.0f, -1.0f},
    };
    gli::texture2d target;
    softRenderer.render(frameSettings, target);
    return EXIT_SUCCESS;
}

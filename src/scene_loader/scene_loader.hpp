#pragma once

#include <scene/fwd.hpp>

#include <QtCore/QDir>
#include <QtCore/QFileInfo>
#include <QtCore/QStringList>

#include <tuple>

#include <scene_loader/scene_loader_export.h>

namespace scene_loader
{
[[nodiscard]] QStringList getSupportedExtensions() SCENE_LOADER_EXPORT;
[[nodiscard]] bool load(scene::Scene & scene, QFileInfo sceneFileInfo) SCENE_LOADER_EXPORT;
[[nodiscard]] bool cachingLoad(scene::Scene & scene, QFileInfo sceneFileInfo, QDir cacheDir) SCENE_LOADER_EXPORT;
}  // namespace scene_loader

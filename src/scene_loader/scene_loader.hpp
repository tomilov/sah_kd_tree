#pragma once

#include <scene/fwd.hpp>

#include <QtCore/QDir>
#include <QtCore/QFileInfo>
#include <QtCore/QStringList>

#include <tuple>

#include <scene_loader/scene_loader_export.h>

namespace scene_loader
{
QStringList getSupportedExtensions() SCENE_LOADER_EXPORT;
bool load(scene::Scene & scene, QFileInfo sceneFileInfo) SCENE_LOADER_EXPORT;
bool cachingLoad(scene::Scene & scene, QFileInfo sceneFileInfo, QDir cacheDir) SCENE_LOADER_EXPORT;
}  // namespace scene_loader

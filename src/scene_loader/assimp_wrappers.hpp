#pragma once

#include <assimp/IOSystem.hpp>
#include <assimp/Logger.hpp>
#include <assimp/ProgressHandler.hpp>

#include <QtCore/QIODevice>
#include <QtCore/QMap>

namespace scene_loader
{
struct AssimpLoggerGuard
{
    explicit AssimpLoggerGuard(Assimp::Logger::LogSeverity logSeverity = Assimp::Logger::LogSeverity::NORMAL);
    ~AssimpLoggerGuard();
};

struct AssimpProgressHandler : Assimp::ProgressHandler
{
    using Assimp::ProgressHandler::ProgressHandler;

    [[nodiscard]] bool Update(float percentage) override;
};

struct AssimpIOSystem : Assimp::IOSystem
{
    AssimpIOSystem();
    ~AssimpIOSystem() override;

    [[nodiscard]] bool Exists(const char * pFile) const override;
    [[nodiscard]] char getOsSeparator() const override;
    [[nodiscard]] Assimp::IOStream * Open(const char * pFile, const char * pMode) override;
    void Close(Assimp::IOStream * pFile) override;

private:
    QMap<QByteArray, QIODevice::OpenMode> openModeMap;
};

}  // namespace scene_loader

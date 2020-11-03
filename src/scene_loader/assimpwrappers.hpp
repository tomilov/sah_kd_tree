#pragma once

#include <assimp/IOSystem.hpp>
#include <assimp/ProgressHandler.hpp>

#include <QtCore>

Q_DECLARE_LOGGING_CATEGORY(assimpWrappers)

struct AssimpLoggerGuard
{
    AssimpLoggerGuard();
    ~AssimpLoggerGuard();
};

struct AssimpProgressHandler : Assimp::ProgressHandler
{
    using Assimp::ProgressHandler::ProgressHandler;

    virtual bool Update(float percentage) override;
};

struct AssimpIOSystem : Assimp::IOSystem
{
    AssimpIOSystem();
    ~AssimpIOSystem() override;

    bool Exists(const char * pFile) const override;
    char getOsSeparator() const override;
    Assimp::IOStream * Open(const char * pFile, const char * pMode) override;
    void Close(Assimp::IOStream * pFile) override;

private:
    QMap<QByteArray, QIODevice::OpenMode> openModeMaps;
};

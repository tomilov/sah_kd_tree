#include <utils/auto_cast.hpp>
#include <viewer/renderer_io.hpp>

#include <QtCore/QDebug>
#include <QtCore/QDir>
#include <QtCore/QFile>
#include <QtCore/QFileInfo>
#include <QtCore/QIODeviceBase>
#include <QtCore/QLoggingCategory>
#include <QtCore/QSaveFile>
#include <QtCore/QStandardPaths>
#include <QtCore/QString>
#include <QtCore/QStringLiteral>

#include <string>
#include <vector>

using namespace std::string_view_literals;

namespace viewer
{
namespace
{
Q_DECLARE_LOGGING_CATEGORY(viewerRendererIoCategory)
Q_LOGGING_CATEGORY(viewerRendererIoCategory, "viewer.renderer_io")
}  // namespace

std::vector<uint8_t> RendererIo::loadPipelineCache(std::string_view pipelineCacheName) const
{
    auto cacheFileName = QString::fromStdString(std::string{pipelineCacheName});
    cacheFileName.append(QStringLiteral(".bin"));
    auto cacheLocations = QStandardPaths::standardLocations(QStandardPaths::StandardLocation::CacheLocation);
    for (const auto & cacheLocation : cacheLocations) {
        QDir cacheDir{cacheLocation};
        if (!cacheDir.exists()) {
            qCWarning(viewerRendererIoCategory).noquote() << QStringLiteral("Possible location '%1' of pipeline cache directory '%2' does not exists").arg(cacheDir.path(), cacheFileName);
            continue;
        }
        QFileInfo cacheFileInfo{cacheDir, cacheFileName};
        if (!cacheFileInfo.exists()) {
            qCWarning(viewerRendererIoCategory).noquote() << QStringLiteral("Possible location '%1' of pipeline cache file '%2' does not exists").arg(cacheFileInfo.path(), cacheFileName);
            continue;
        }
        QFile cacheFile{cacheFileInfo.filePath()};
        if (!cacheFile.open(QIODeviceBase::OpenModeFlag::ReadOnly)) {
            qCWarning(viewerRendererIoCategory).noquote() << QStringLiteral("Possible location '%1' of pipeline cache file '%2' cannot be opened to read: %3").arg(cacheFile.fileName(), cacheFileName, cacheFile.errorString());
            continue;
        }
        auto cacheFileSize = std::size(cacheFile);
        size_t dataSize = utils::autoCast(cacheFileSize);
        std::vector<uint8_t> cacheData(dataSize);
        qint64 bytesRead = cacheFile.read(utils::autoCast(std::data(cacheData)), cacheFileSize);
        if (bytesRead < 0) {
            qCWarning(viewerRendererIoCategory).noquote() << QStringLiteral("Failed to read pipeline cache data '%1' from file '%2': %3").arg(cacheFileName, cacheFile.fileName(), cacheFile.errorString());
            continue;
        }
        if (bytesRead != cacheFileSize) {
            qCWarning(viewerRendererIoCategory).noquote()
                << QStringLiteral("Failed to read pipeline cache data '%1' from file '%2': bytes read %4 != file size %5: %3").arg(cacheFileName, cacheFile.fileName(), cacheFile.errorString()).arg(bytesRead).arg(cacheFileSize);
            continue;
        }
        qCInfo(viewerRendererIoCategory).noquote() << QStringLiteral("Pipeline cache data '%1' successfully read from file '%2'").arg(cacheFileName, cacheFile.fileName());
        return cacheData;
    }
    qCWarning(viewerRendererIoCategory).noquote() << QStringLiteral("Failed to load pipeline cache from '%1' location").arg(cacheFileName);
    return {};
}

bool RendererIo::savePipelineCache(const std::vector<uint8_t> & data, std::string_view pipelineCacheName) const
{
    auto cacheFileName = QString::fromStdString(std::string{pipelineCacheName});
    cacheFileName.append(QStringLiteral(".bin"));
    auto cacheLocation = QStandardPaths::writableLocation(QStandardPaths::StandardLocation::CacheLocation);
    QDir cacheDir{cacheLocation};
    if (!cacheDir.exists()) {
        qCWarning(viewerRendererIoCategory).noquote() << QStringLiteral("Possible location '%1' of pipeline cache directory '%2' does not exists").arg(cacheDir.path(), cacheFileName);
        return false;
    }
    QFileInfo cacheFileInfo{cacheDir, cacheFileName};
    QSaveFile cacheFile{cacheFileInfo.filePath()};
    if (!cacheFile.open(QIODeviceBase::OpenModeFlag::WriteOnly)) {
        qCWarning(viewerRendererIoCategory).noquote() << QStringLiteral("Possible location '%1' of pipeline cache file '%2' cannot be opened to write: %3").arg(cacheFile.fileName(), cacheFileName, cacheFile.errorString());
        return false;
    }
    size_t dataSize = std::size(data);
    qint64 cacheDataSize = utils::autoCast(dataSize);
    qint64 bytesWritten = cacheFile.write(utils::autoCast(std::data(data)), cacheDataSize);
    if (bytesWritten < 0) {
        qCWarning(viewerRendererIoCategory).noquote() << QStringLiteral("Failed to write pipeline cache data '%1' to file '%2': %3").arg(cacheFileName, cacheFile.fileName(), cacheFile.errorString());
        return false;
    }
    if (bytesWritten != cacheDataSize) {
        qCWarning(viewerRendererIoCategory).noquote()
            << QStringLiteral("Failed to write pipeline cache data '%1' to file '%2': bytes written %4 != file size %5: %3").arg(cacheFileName, cacheFile.fileName(), cacheFile.errorString()).arg(bytesWritten).arg(cacheDataSize);
        return false;
    }
    if (!cacheFile.commit()) {
        qCWarning(viewerRendererIoCategory).noquote() << QStringLiteral("Failed to commit writing of pipeline cache data '%1' to file '%2'").arg(cacheFileName, cacheFile.fileName(), cacheFile.errorString());
        return false;
    }
    qCInfo(viewerRendererIoCategory).noquote() << QStringLiteral("Pipeline cache data '%1' successfully written to file '%2'").arg(cacheFileName, cacheFile.fileName());
    return true;
}

std::vector<uint32_t> RendererIo::loadShader(std::string_view shaderName) const
{
    auto shaderFileName = QString::fromStdString(std::string{shaderName});
    shaderFileName.append(QStringLiteral(".spv"));
    auto shaderLocation = QStringLiteral(":/shaders");
    QDir shaderDir{shaderLocation};
    if (!shaderDir.exists()) {
        qCWarning(viewerRendererIoCategory).noquote() << QStringLiteral("Possible location '%1' of shader directory '%2' does not exists").arg(shaderDir.path(), shaderFileName);
        return {};
    }
    QFileInfo shaderFileInfo{shaderDir, shaderFileName};
    if (!shaderFileInfo.exists()) {
        qCWarning(viewerRendererIoCategory).noquote() << QStringLiteral("Possible location '%1' of shader file '%2' does not exists").arg(shaderFileInfo.path(), shaderFileName);
        return {};
    }
    QFile shaderFile{shaderFileInfo.filePath()};
    if (!shaderFile.open(QIODeviceBase::OpenModeFlag::ReadOnly)) {
        qCWarning(viewerRendererIoCategory).noquote() << QStringLiteral("Possible location '%1' of shader file '%2' cannot be opened to read: %3").arg(shaderFile.fileName(), shaderFileName, shaderFile.errorString());
        return {};
    }
    auto shaderFileSize = std::size(shaderFile);
    size_t dataSize = utils::autoCast(shaderFileSize);
    std::vector<uint32_t> shaderData(dataSize);
    qint64 bytesRead = shaderFile.read(utils::autoCast(std::data(shaderData)), shaderFileSize);
    if (bytesRead < 0) {
        qCWarning(viewerRendererIoCategory).noquote() << QStringLiteral("Failed to read shader data '%1' from file '%2': %3").arg(shaderFileName, shaderFile.fileName(), shaderFile.errorString());
        return {};
    }
    if (bytesRead != shaderFileSize) {
        qCWarning(viewerRendererIoCategory).noquote()
            << QStringLiteral("Failed to read shader data '%1' from file '%2': bytes read %4 != file size %5: %3").arg(shaderFileName, shaderFile.fileName(), shaderFile.errorString()).arg(bytesRead).arg(shaderFileSize);
        return {};
    }
    qCInfo(viewerRendererIoCategory).noquote() << QStringLiteral("Shader data '%1' successfully read from file '%2'").arg(shaderFileName, shaderFile.fileName());
    return shaderData;
}

}  // namespace viewer

#include <utils/assert.hpp>
#include <utils/auto_cast.hpp>
#include <viewer/file_io.hpp>

#include <QtCore/QDebug>
#include <QtCore/QDir>
#include <QtCore/QFile>
#include <QtCore/QFileInfo>
#include <QtCore/QIODeviceBase>
#include <QtCore/QLoggingCategory>
#include <QtCore/QSaveFile>
#include <QtCore/QStandardPaths>
#include <QtCore/QString>

#include <string>
#include <vector>

#include <cstddef>
#include <cstdint>

using namespace std::string_view_literals;
using namespace Qt::StringLiterals;

namespace viewer
{
namespace
{
Q_DECLARE_LOGGING_CATEGORY(viewerFileIoCategory)
Q_LOGGING_CATEGORY(viewerFileIoCategory, "viewer.file_io")
}  // namespace

FileIo::FileIo(QString shaderLocation) : shaderLocation{shaderLocation}
{}

std::vector<uint8_t> FileIo::loadPipelineCache(std::string_view pipelineCacheName) const
{
    auto cacheFileName = QString::fromStdString(std::string{pipelineCacheName});
    cacheFileName.append(u".bin"_s);
    auto cacheLocations = QStandardPaths::standardLocations(QStandardPaths::StandardLocation::CacheLocation);
    for (const auto & cacheLocation : cacheLocations) {
        QDir cacheDir{cacheLocation};
        if (!cacheDir.exists()) {
            qCWarning(viewerFileIoCategory).noquote() << u"Possible location '%1' of pipeline cache directory '%2' does not exists"_s.arg(cacheDir.path(), cacheFileName);
            continue;
        }
        QFileInfo cacheFileInfo{cacheDir, cacheFileName};
        if (!cacheFileInfo.exists()) {
            qCWarning(viewerFileIoCategory).noquote() << u"Possible location '%1' of pipeline cache file '%2' does not exists"_s.arg(cacheFileInfo.path(), cacheFileName);
            continue;
        }
        QFile cacheFile{cacheFileInfo.filePath()};
        if (!cacheFile.open(QIODeviceBase::OpenModeFlag::ReadOnly)) {
            qCWarning(viewerFileIoCategory).noquote() << u"Possible location '%1' of pipeline cache file '%2' cannot be opened to read: %3"_s.arg(cacheFile.fileName(), cacheFileName, cacheFile.errorString());
            continue;
        }
        auto cacheFileSize = cacheFile.size();
        size_t dataSize = utils::autoCast(cacheFileSize);
        std::vector<uint8_t> cacheData(dataSize);
        qint64 bytesRead = cacheFile.read(utils::autoCast(std::data(cacheData)), cacheFileSize);
        if (bytesRead < 0) {
            qCWarning(viewerFileIoCategory).noquote() << u"Failed to read pipeline cache data '%1' from file '%2': %3"_s.arg(cacheFileName, cacheFile.fileName(), cacheFile.errorString());
            continue;
        }
        if (bytesRead != cacheFileSize) {
            qCWarning(viewerFileIoCategory).noquote()
                << u"Failed to read pipeline cache data '%1' from file '%2': bytes read %4 != file size %5: %3"_s.arg(cacheFileName, cacheFile.fileName(), cacheFile.errorString()).arg(bytesRead).arg(cacheFileSize);
            continue;
        }
        qCInfo(viewerFileIoCategory).noquote() << u"Pipeline cache data file '%1' successfully read from '%2'"_s.arg(cacheFileName, cacheDir.path());
        return cacheData;
    }
    qCWarning(viewerFileIoCategory).noquote() << u"Failed to load pipeline cache from '%1' location"_s.arg(cacheFileName);
    return {};
}

bool FileIo::savePipelineCache(const std::vector<uint8_t> & data, std::string_view pipelineCacheName) const
{
    auto cacheFileName = QString::fromStdString(std::string{pipelineCacheName});
    cacheFileName.append(u".bin"_s);
    auto cacheLocation = QStandardPaths::writableLocation(QStandardPaths::StandardLocation::CacheLocation);
    QDir cacheDir{cacheLocation};
    if (!cacheDir.exists()) {
        qCWarning(viewerFileIoCategory).noquote() << u"Possible location '%1' of pipeline cache directory '%2' does not exists"_s.arg(cacheDir.path(), cacheFileName);
        return false;
    }
    QFileInfo cacheFileInfo{cacheDir, cacheFileName};
    QSaveFile cacheFile{cacheFileInfo.filePath()};
    if (!cacheFile.open(QIODeviceBase::OpenModeFlag::WriteOnly)) {
        qCWarning(viewerFileIoCategory).noquote() << u"Possible location '%1' of pipeline cache file '%2' cannot be opened to write: %3"_s.arg(cacheFile.fileName(), cacheFileName, cacheFile.errorString());
        return false;
    }
    size_t dataSize = std::size(data);
    qint64 cacheDataSize = utils::autoCast(dataSize);
    qint64 bytesWritten = cacheFile.write(utils::autoCast(std::data(data)), cacheDataSize);
    if (bytesWritten < 0) {
        qCWarning(viewerFileIoCategory).noquote() << u"Failed to write pipeline cache data '%1' to file '%2': %3"_s.arg(cacheFileName, cacheFile.fileName(), cacheFile.errorString());
        return false;
    }
    if (bytesWritten != cacheDataSize) {
        qCWarning(viewerFileIoCategory).noquote()
            << u"Failed to write pipeline cache data '%1' to file '%2': bytes written %4 != file size %5: %3"_s.arg(cacheFileName, cacheFile.fileName(), cacheFile.errorString()).arg(bytesWritten).arg(cacheDataSize);
        return false;
    }
    if (!cacheFile.commit()) {
        qCWarning(viewerFileIoCategory).noquote() << u"Failed to commit writing of pipeline cache data '%1' to file '%2': %3"_s.arg(cacheFileName, cacheFile.fileName(), cacheFile.errorString());
        return false;
    }
    qCInfo(viewerFileIoCategory).noquote() << u"Pipeline cache data file '%1' successfully written to '%2'"_s.arg(cacheFileName, cacheDir.path());
    return true;
}

std::vector<uint32_t> FileIo::loadShader(std::string_view shaderName) const
{
    auto shaderFileName = QString::fromStdString(std::string{shaderName});
    shaderFileName.append(u".spv"_s);
    QDir shaderDir{shaderLocation};
    if (!shaderDir.exists()) {
        qCWarning(viewerFileIoCategory).noquote() << u"Possible location '%1' of shader directory '%2' does not exists"_s.arg(shaderDir.path(), shaderFileName);
        return {};
    }
    QFileInfo shaderFileInfo{shaderDir, shaderFileName};
    if (!shaderFileInfo.exists()) {
        qCWarning(viewerFileIoCategory).noquote() << u"Possible location '%1' of shader file '%2' does not exists"_s.arg(shaderFileInfo.path(), shaderFileName);
        return {};
    }
    QFile shaderFile{shaderFileInfo.filePath()};
    if (!shaderFile.open(QIODeviceBase::OpenModeFlag::ReadOnly)) {
        qCWarning(viewerFileIoCategory).noquote() << u"Possible location '%1' of shader file '%2' cannot be opened to read: %3"_s.arg(shaderFile.fileName(), shaderFileName, shaderFile.errorString());
        return {};
    }
    auto shaderFileSize = shaderFile.size();
    size_t dataSize = utils::autoCast(shaderFileSize);
    std::vector<uint32_t> spirv;
    INVARIANT((dataSize % sizeof *std::data(spirv)) == 0, "Expected whole number of double words for SPIR-V");
    spirv.resize(dataSize / sizeof *std::data(spirv));
    qint64 bytesRead = shaderFile.read(utils::autoCast(std::data(spirv)), shaderFileSize);
    if (bytesRead < 0) {
        qCWarning(viewerFileIoCategory).noquote() << u"Failed to read shader data '%1' from file '%2': %3"_s.arg(shaderFileName, shaderFile.fileName(), shaderFile.errorString());
        return {};
    }
    if (bytesRead != shaderFileSize) {
        qCWarning(viewerFileIoCategory).noquote() << u"Failed to read shader data '%1' from file '%2': bytes read %4 != file size %5: %3"_s.arg(shaderFileName, shaderFile.fileName(), shaderFile.errorString()).arg(bytesRead).arg(shaderFileSize);
        return {};
    }
    qCInfo(viewerFileIoCategory).noquote() << u"Shader data '%1' successfully read from file '%2'"_s.arg(shaderFileName, shaderFile.fileName());
    return spirv;
}

}  // namespace viewer

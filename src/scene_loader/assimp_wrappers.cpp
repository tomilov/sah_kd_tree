#include <scene_loader/assimp_wrappers.hpp>

#include <assimp/DefaultLogger.hpp>
#include <assimp/IOStream.hpp>
#include <assimp/Logger.hpp>

#include <QtCore/QByteArray>
#include <QtCore/QDebug>
#include <QtCore/QDir>
#include <QtCore/QFile>
#include <QtCore/QFileDevice>
#include <QtCore/QFileInfo>
#include <QtCore/QLoggingCategory>
#include <QtCore/QString>
#include <QtCore/QStringLiteral>

#include <memory>

#include <cstddef>

namespace scene_loader
{
namespace
{
Q_DECLARE_LOGGING_CATEGORY(assimpWrappersLog)
Q_LOGGING_CATEGORY(assimpWrappersLog, "scene_loader.assimp")
}  // namespace

namespace
{

struct AssimpLogger : Assimp::Logger
{
    using Assimp::Logger::Logger;

    virtual bool attachStream(Assimp::LogStream * pStream, unsigned int severity) override;
    virtual bool detachStream(Assimp::LogStream * pStream, unsigned int severity) override;

private:
    friend AssimpLoggerGuard;

    virtual void OnVerboseDebug(const char * message) override;
    virtual void OnDebug(const char * message) override;
    virtual void OnInfo(const char * message) override;
    virtual void OnWarn(const char * message) override;
    virtual void OnError(const char * message) override;
};

struct AssimpIOStream : Assimp::IOStream
{
    AssimpIOStream(QIODevice * device);
    ~AssimpIOStream() override;

    size_t Read(void * pvBuffer, size_t pSize, size_t pCount) override;
    size_t Write(const void * pvBuffer, size_t pSize, size_t pCount) override;
    aiReturn Seek(size_t pOffset, aiOrigin pOrigin) override;
    size_t Tell() const override;
    size_t FileSize() const override;
    void Flush() override;

private:
    const std::unique_ptr<QIODevice> device;
};

bool AssimpLogger::attachStream(Assimp::LogStream * pStream, unsigned int severity)
{
    Q_UNUSED(pStream);
    Q_UNUSED(severity);
    return true;
}

bool AssimpLogger::detachStream(Assimp::LogStream * pStream, unsigned int severity)
{
    Q_UNUSED(pStream);
    Q_UNUSED(severity);
    return true;
}

void AssimpLogger::OnVerboseDebug(const char * message)
{
    qCDebug(assimpWrappersLog) << message;
}

void AssimpLogger::OnDebug(const char * message)
{
    qCDebug(assimpWrappersLog) << message;
}

void AssimpLogger::OnInfo(const char * message)
{
    qCInfo(assimpWrappersLog) << message;
}

void AssimpLogger::OnWarn(const char * message)
{
    qCWarning(assimpWrappersLog) << message;
}

void AssimpLogger::OnError(const char * message)
{
    qCCritical(assimpWrappersLog) << message;
}

}  // namespace

AssimpLoggerGuard::AssimpLoggerGuard(Assimp::Logger::LogSeverity logSeverity)
{
    Assimp::DefaultLogger::set(new AssimpLogger{logSeverity});
}  // NOLINT(clang-analyzer-cplusplus.NewDeleteLeaks)

AssimpLoggerGuard::~AssimpLoggerGuard()
{
    Assimp::DefaultLogger::kill();
}

bool AssimpProgressHandler::Update(float percentage)
{
    qCInfo(assimpWrappersLog).noquote() << QStringLiteral("%1 loaded").arg(qreal(percentage));
    return true;
}

AssimpIOStream::AssimpIOStream(QIODevice * device) : device{device}
{}

AssimpIOStream::~AssimpIOStream() = default;

size_t AssimpIOStream::Read(void * pvBuffer, size_t pSize, size_t pCount)
{
    auto readBytes = device->read(static_cast<char *>(pvBuffer), qint64(pSize * pCount));
    if (readBytes < 0) {
        qCWarning(assimpWrappersLog) << "reading failed";
    }
    return size_t(readBytes);
}

size_t AssimpIOStream::Write(const void * pvBuffer, size_t pSize, size_t pCount)
{
    auto writtenBytes = device->write(static_cast<const char *>(pvBuffer), qint64(pSize * pCount));
    if (writtenBytes < 0) {
        qCWarning(assimpWrappersLog) << "writing failed";
    }
    return size_t(writtenBytes);
}

aiReturn AssimpIOStream::Seek(size_t pOffset, aiOrigin pOrigin)
{
    auto seekPos = qint64(pOffset);

    if (pOrigin == aiOrigin_CUR) {
        seekPos += device->pos();
    } else if (pOrigin == aiOrigin_END) {
        seekPos += device->size();
    }

    if (!device->seek(seekPos)) {
        qCWarning(assimpWrappersLog) << "seeking failed";
        return aiReturn_FAILURE;
    }
    return aiReturn_SUCCESS;
}

size_t AssimpIOStream::Tell() const
{
    return size_t(device->pos());
}

size_t AssimpIOStream::FileSize() const
{
    return size_t(device->size());
}

void AssimpIOStream::Flush()
{
    if (auto file = qobject_cast<QFileDevice *>(device.get())) {
        file->flush();
    }
}

AssimpIOSystem::AssimpIOSystem()
{
    openModeMap["r"] = QIODevice::ReadOnly;
    openModeMap["rb"] = QIODevice::ReadOnly;
    openModeMap["rt"] = QIODevice::ReadOnly | QIODevice::Text;
    openModeMap["r+"] = QIODevice::ReadWrite;
    openModeMap["w+"] = QIODevice::ReadWrite | QIODevice::Truncate;
    openModeMap["a+"] = QIODevice::ReadWrite | QIODevice::Append;
    openModeMap["wb"] = QIODevice::WriteOnly;
    openModeMap["w"] = QIODevice::WriteOnly | QIODevice::Truncate;
    openModeMap["a"] = QIODevice::WriteOnly | QIODevice::Append;
    openModeMap["wt"] = QIODevice::WriteOnly | QIODevice::Text;
}

AssimpIOSystem::~AssimpIOSystem() = default;

bool AssimpIOSystem::Exists(const char * pFile) const
{
    return QFileInfo::exists(QString::fromUtf8(pFile));
}

char AssimpIOSystem::getOsSeparator() const
{
    return QDir::separator().toLatin1();
}

Assimp::IOStream * AssimpIOSystem::Open(const char * pFile, const char * pMode)
{
    const QString fileName{QString::fromUtf8(pFile)};
    const QByteArray cleanedMode{QByteArray(pMode).trimmed()};

    const QIODevice::OpenMode openMode = openModeMap.value(cleanedMode, QIODevice::NotOpen);

    std::unique_ptr<QFile> file{new QFile(fileName)};
    if (!file->open(openMode)) {
        return {};
    }
    return new AssimpIOStream{file.release()};
}

void AssimpIOSystem::Close(Assimp::IOStream * pFile)
{
    Q_ASSERT(dynamic_cast<AssimpIOStream *>(pFile));
    delete pFile;
}

}  // namespace scene_loader

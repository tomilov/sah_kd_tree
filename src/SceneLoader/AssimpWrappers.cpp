#include "AssimpWrappers.hpp"

#include <assimp/DefaultLogger.hpp>
#include <assimp/IOStream.hpp>
#include <assimp/Logger.hpp>

Q_LOGGING_CATEGORY(assimpWrappers, "assimpWrappers")

struct Logger : Assimp::Logger
{
    using Assimp::Logger::Logger;

    virtual bool attachStream(Assimp::LogStream * pStream, unsigned int severity) override;
    virtual bool detatchStream(Assimp::LogStream * pStream, unsigned int severity) override;

private:
    friend AssimpLoggerGuard;

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
    QScopedPointer<QIODevice> const device;
};

bool Logger::attachStream(Assimp::LogStream * pStream, unsigned int severity)
{
    Q_UNUSED(pStream);
    Q_UNUSED(severity);
    return true;
}

bool Logger::detatchStream(Assimp::LogStream * pStream, unsigned int severity)
{
    Q_UNUSED(pStream);
    Q_UNUSED(severity);
    return true;
}

void Logger::OnDebug(const char * message)
{
    qCDebug(assimpWrappers) << message;
}

void Logger::OnInfo(const char * message)
{
    qCInfo(assimpWrappers) << message;
}

void Logger::OnWarn(const char * message)
{
    qCWarning(assimpWrappers) << message;
}

void Logger::OnError(const char * message)
{
    qCCritical(assimpWrappers) << message;
}

AssimpLoggerGuard::AssimpLoggerGuard(Assimp::Logger::LogSeverity logSeverity)
{
    Assimp::DefaultLogger::set(new Logger{logSeverity});
}

AssimpLoggerGuard::~AssimpLoggerGuard()
{
    Assimp::DefaultLogger::kill();
}

bool AssimpProgressHandler::Update(float percentage)
{
    qCInfo(assimpWrappers) << QStringLiteral("%1 loaded").arg(qreal(percentage));
    return true;
}

AssimpIOStream::AssimpIOStream(QIODevice * device)
    : device{device}
{}

AssimpIOStream::~AssimpIOStream() = default;

size_t AssimpIOStream::Read(void * pvBuffer, size_t pSize, size_t pCount)
{
    auto readBytes = device->read(static_cast<char *>(pvBuffer), qint64(pSize * pCount));
    if (readBytes < 0) {
        qCWarning(assimpWrappers) << QStringLiteral("reading failed");
    }
    return size_t(readBytes);
}

size_t AssimpIOStream::Write(const void * pvBuffer, size_t pSize, size_t pCount)
{
    auto writtenBytes = device->write(static_cast<const char *>(pvBuffer), qint64(pSize * pCount));
    if (writtenBytes < 0) {
        qCWarning(assimpWrappers) << QStringLiteral("writing failed");
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
        qCWarning(assimpWrappers) << QStringLiteral("seeking failed");
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
    openModeMap[QByteArrayLiteral("r")] = QIODevice::ReadOnly;
    openModeMap[QByteArrayLiteral("rb")] = QIODevice::ReadOnly;
    openModeMap[QByteArrayLiteral("rt")] = QIODevice::ReadOnly | QIODevice::Text;
    openModeMap[QByteArrayLiteral("r+")] = QIODevice::ReadWrite;
    openModeMap[QByteArrayLiteral("w+")] = QIODevice::ReadWrite | QIODevice::Truncate;
    openModeMap[QByteArrayLiteral("a+")] = QIODevice::ReadWrite | QIODevice::Append;
    openModeMap[QByteArrayLiteral("wb")] = QIODevice::WriteOnly;
    openModeMap[QByteArrayLiteral("w")] = QIODevice::WriteOnly | QIODevice::Truncate;
    openModeMap[QByteArrayLiteral("a")] = QIODevice::WriteOnly | QIODevice::Append;
    openModeMap[QByteArrayLiteral("wt")] = QIODevice::WriteOnly | QIODevice::Text;
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

    QScopedPointer<QFile> file{new QFile(fileName)};
    if (!file->open(openMode)) {
        return {};
    }
    return new AssimpIOStream{file.take()};
}

void AssimpIOSystem::Close(Assimp::IOStream * pFile)
{
    Q_ASSERT(dynamic_cast<AssimpIOStream *>(pFile));
    delete pFile;
}

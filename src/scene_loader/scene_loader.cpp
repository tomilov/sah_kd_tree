#include <scene_loader/assimp_wrappers.hpp>
#include <scene_loader/scene_loader.hpp>

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include <QtCore/QCryptographicHash>
#include <QtCore/QDataStream>
#include <QtCore/QDebug>
#include <QtCore/QDir>
#include <QtCore/QElapsedTimer>
#include <QtCore/QFile>
#include <QtCore/QFileInfo>
#include <QtCore/QLoggingCategory>
#include <QtCore/QString>

namespace scene_loader
{
Q_DECLARE_LOGGING_CATEGORY(sceneLoaderLog)
Q_LOGGING_CATEGORY(sceneLoaderLog, "sceneLoader")

namespace
{
template<typename Type>
QString toString(const Type & value)
{
    QString string;
    QDebug{&string}.noquote().nospace() << value;
    return string;
}

using LoggingCategory = const QLoggingCategory & (*)();

bool checkDataStreamStatus(QDataStream & dataStream, const LoggingCategory loggingCategory, QString description)
{
    const auto status = dataStream.status();
    if (status != QDataStream::Ok) {
        qCWarning(loggingCategory).noquote() << QStringLiteral("%1 (data stream error: %2 %3)").arg(description, toString(status), dataStream.device()->errorString());
        return false;
    }
    return true;
}
}  // namespace

bool SceneLoader::load(QFileInfo sceneFileInfo)
{
    AssimpLoggerGuard loggerGuard{Assimp::Logger::LogSeverity::VERBOSE};
    Assimp::Importer importer;
    importer.SetProgressHandler(new AssimpProgressHandler);
    importer.SetIOHandler(new AssimpIOSystem);

    unsigned int pFlags = aiProcess_Triangulate /*  | aiProcess_ImproveCacheLocality| aiProcess_JoinIdenticalVertices | aiProcess_OptimizeMeshes | aiProcess_OptimizeGraph*/;
    // pFlags |= aiProcess_ValidateDataStructure | aiProcess_FindInvalidData;
    {
        // pFlags |= aiProcess_FindDegenerates;
        pFlags |= aiProcess_SortByPType;
        importer.SetPropertyInteger(AI_CONFIG_PP_SBP_REMOVE, aiPrimitiveType_POINT | aiPrimitiveType_LINE);
        importer.SetPropertyBool(AI_CONFIG_PP_FD_CHECKAREA, false);
    }
    {
        pFlags |= aiProcess_RemoveComponent;
        // all except aiComponent_MESHES
        auto excludeAllComponents = aiComponent_NORMALS | aiComponent_TANGENTS_AND_BITANGENTS | aiComponent_COLORS | aiComponent_TEXCOORDS | aiComponent_BONEWEIGHTS | aiComponent_ANIMATIONS | aiComponent_TEXTURES | aiComponent_LIGHTS |
                                    aiComponent_CAMERAS | aiComponent_MATERIALS;
        importer.SetPropertyInteger(AI_CONFIG_PP_RVC_FLAGS, excludeAllComponents);
    }
    QElapsedTimer sceneLoadTimer;
    sceneLoadTimer.start();
    Q_ASSERT(importer.ValidateFlags(pFlags));
    const auto scene = importer.ReadFile(qPrintable(sceneFileInfo.filePath()), pFlags);
    qCDebug(sceneLoaderLog).noquote() << QStringLiteral("scene loaded in %1 ms").arg(sceneLoadTimer.nsecsElapsed() * 1E-6);
    if (!scene) {
        qCCritical(sceneLoaderLog).noquote() << QStringLiteral("unable to load scene %1: %2").arg(sceneFileInfo.filePath(), QString::fromLocal8Bit(importer.GetErrorString()));
        return false;
    }
    {
        aiMemoryInfo memoryInfo;
        importer.GetMemoryRequirements(memoryInfo);
        qCDebug(sceneLoaderLog).noquote() << QStringLiteral("scene memory info (%1 total): textures %2, materials %3, meshes %4, nodes %5, animations %6, cameras %7, lights %8")
                                                 .arg(memoryInfo.total)
                                                 .arg(memoryInfo.textures)
                                                 .arg(memoryInfo.materials)
                                                 .arg(memoryInfo.meshes)
                                                 .arg(memoryInfo.nodes)
                                                 .arg(memoryInfo.animations)
                                                 .arg(memoryInfo.cameras)
                                                 .arg(memoryInfo.lights);
    }

    qCInfo(sceneLoaderLog) << "scene has animations:" << scene->HasAnimations();
    qCInfo(sceneLoaderLog) << "scene has cameras:" << scene->HasCameras();
    qCInfo(sceneLoaderLog) << "scene has lights:" << scene->HasLights();
    qCInfo(sceneLoaderLog) << "scene has materials (required):" << scene->HasMaterials();
    qCInfo(sceneLoaderLog) << "scene has meshes (required):" << scene->HasMeshes();
    qCInfo(sceneLoaderLog) << "scene has textures:" << scene->HasTextures();
    if (!scene->HasMeshes()) {
        qCCritical(sceneLoaderLog).noquote() << QStringLiteral("Scene %1 is empty").arg(sceneFileInfo.filePath());
        return false;
    }

    qCInfo(sceneLoaderLog).noquote() << QStringLiteral("scene flags: %1").arg(scene->mFlags);
    qCInfo(sceneLoaderLog).noquote() << QStringLiteral("number of animations: %1").arg(scene->mNumAnimations);
    qCInfo(sceneLoaderLog).noquote() << QStringLiteral("number of cameras: %1").arg(scene->mNumCameras);
    qCInfo(sceneLoaderLog).noquote() << QStringLiteral("number of lights: %1").arg(scene->mNumLights);
    qCInfo(sceneLoaderLog).noquote() << QStringLiteral("number of materials: %1").arg(scene->mNumMaterials);
    qCInfo(sceneLoaderLog).noquote() << QStringLiteral("number of meshes: %1").arg(scene->mNumMeshes);
    qCInfo(sceneLoaderLog).noquote() << QStringLiteral("number of textures: %1").arg(scene->mNumTextures);

    unsigned int numVertices = 0;
    unsigned int numFaces = 0;
    for (unsigned int i = 0; i < scene->mNumMeshes; ++i) {
        numVertices += scene->mMeshes[i]->mNumVertices;
        numFaces += scene->mMeshes[i]->mNumFaces;
    }
    qCInfo(sceneLoaderLog).noquote() << QStringLiteral("total number of vertices: %1").arg(numVertices);
    qCInfo(sceneLoaderLog).noquote() << QStringLiteral("total number of faces: %1").arg(numFaces);

    auto triangleCount = qint32(numFaces);
    triangles.resize(int(triangleCount));
    {
        auto t = triangles.data();
        const auto toVertex = [](const aiVector3D & v) -> Vertex { return {v.x, v.y, v.z}; };
        for (unsigned int m = 0; m < scene->mNumMeshes; ++m) {
            const aiMesh & mesh = *scene->mMeshes[m];
            // qCDebug(sceneLoader) << mesh.mName.C_Str();
            if ((mesh.mPrimitiveTypes & ~(aiPrimitiveType_TRIANGLE | aiPrimitiveType_NGONEncodingFlag)) != 0) {
                qCWarning(sceneLoaderLog).noquote() << QStringLiteral("mesh %1 contains not only triangles (possibly NGON-encoded)").arg(m);
                return false;
            }
            for (unsigned int f = 0; f < mesh.mNumFaces; ++f) {
                const aiFace & face = mesh.mFaces[f];
                if (face.mNumIndices != 3) {
                    qCWarning(sceneLoaderLog).noquote() << QStringLiteral("number of vertices %1 is not equal to 3, face %2 of mesh %3 is not a triangle").arg(numVertices).arg(f).arg(m);
                    return false;
                }
                auto A = toVertex(mesh.mVertices[face.mIndices[0]]);
                auto B = toVertex(mesh.mVertices[face.mIndices[1]]);
                auto C = toVertex(mesh.mVertices[face.mIndices[2]]);
                *t++ = {A, B, C};
            }
        }
        Q_ASSERT(triangles.data() + triangleCount == t);
    }
    return true;
}

QFileInfo SceneLoader::getCacheEntryFileInfo(QFileInfo sceneFileInfo, QDir cacheDir)
{
    QFile sceneFile{sceneFileInfo.filePath()};
    if (!sceneFile.open(QFile::ReadOnly)) {
        return {};
    }
    QCryptographicHash cryptographicHash{QCryptographicHash::Algorithm::Md5};
    cryptographicHash.addData(sceneFileInfo.filePath().toUtf8());
    if (!cryptographicHash.addData(&sceneFile)) {
        return {};
    }
    QFileInfo cacheEntryFileInfo;
    cacheEntryFileInfo.setFile(cacheDir, QString::fromUtf8(cryptographicHash.result().toHex()).append(".triangle"));
    return cacheEntryFileInfo;
}

bool SceneLoader::loadFromCache(QFileInfo cacheEntryFileInfo)
{
    QElapsedTimer loadTimer;
    loadTimer.start();
    QFile cacheEntryFile{cacheEntryFileInfo.filePath()};
    if (!cacheEntryFile.open(QFile::ReadOnly)) {
        qCCritical(sceneLoaderLog).noquote() << QStringLiteral("unable to open file %1 to read triangle from: %2").arg(cacheEntryFile.fileName(), toString(cacheEntryFile.error()));
        return false;
    }
    qCInfo(sceneLoaderLog).noquote() << QStringLiteral("start to load triangles from file %1").arg(cacheEntryFile.fileName());
    QDataStream dataStream{&cacheEntryFile};
    qint32 triangleCount = 0;
    if (!checkDataStreamStatus(dataStream >> triangleCount, sceneLoaderLog, QStringLiteral("unable to read count of triangles to file %1").arg(cacheEntryFile.fileName()))) {
        return false;
    }
    triangles.resize(int(triangleCount));
    const int len = int(triangles.size() * sizeof *triangles.data());
    const int readLen = dataStream.readRawData(reinterpret_cast<char *>(triangles.data()), len);
    if (readLen != len) {
        qCInfo(sceneLoaderLog).noquote() << QStringLiteral("unable to read triangles from file %1: need %2 bytes, read %3 bytes").arg(cacheEntryFile.fileName()).arg(len).arg(readLen);
        return false;
    }
    if (!checkDataStreamStatus(dataStream, sceneLoaderLog, QStringLiteral("unable to read triangles from file %1").arg(cacheEntryFile.fileName()))) {
        return false;
    }
    qCInfo(sceneLoaderLog).noquote() << QStringLiteral("%3 ms to load %1 triangles from file %2").arg(triangles.size()).arg(cacheEntryFile.fileName()).arg(loadTimer.nsecsElapsed() * 1E-6);
    return true;
}

bool SceneLoader::storeToCache(QFileInfo cacheEntryFileInfo)
{
    QElapsedTimer saveTimer;
    saveTimer.start();
    QFile cacheEntryFile{cacheEntryFileInfo.filePath()};
    if (!cacheEntryFile.open(QFile::NewOnly)) {
        qCCritical(sceneLoaderLog).noquote() << QStringLiteral("unable to open file %1 to write triangles to: %2").arg(cacheEntryFile.fileName(), toString(cacheEntryFile.error()));
        return false;
    }
    qCInfo(sceneLoaderLog).noquote() << QStringLiteral("start to save triangles to file %1").arg(cacheEntryFile.fileName());
    QDataStream dataStream{&cacheEntryFile};
    auto triangleCount = qint32(triangles.size());
    if (!checkDataStreamStatus(dataStream << triangleCount, sceneLoaderLog, QStringLiteral("unable to write count of triangles to file %1").arg(cacheEntryFile.fileName()))) {
        return false;
    }
    const int len = int(triangles.size() * sizeof *triangles.data());
    const int writeLen = dataStream.writeRawData(reinterpret_cast<const char *>(triangles.data()), len);
    if (len != writeLen) {
        qCInfo(sceneLoaderLog).noquote() << QStringLiteral("unable to write triangles to file %1: want %2 bytes, but written %3 bytes").arg(cacheEntryFile.fileName()).arg(len).arg(writeLen);
        return false;
    }
    if (!checkDataStreamStatus(dataStream, sceneLoaderLog, QStringLiteral("unable to write triangles to file %1").arg(cacheEntryFile.fileName()))) {
        return false;
    }
    qCInfo(sceneLoaderLog).noquote() << QStringLiteral("%1 triangles successfuly saved to file %2 in %3 ms").arg(triangles.size()).arg(cacheEntryFile.fileName()).arg(saveTimer.nsecsElapsed() * 1E-6);
    return true;
}

bool SceneLoader::cachingLoad(QFileInfo sceneFileInfo, QDir cacheDir)
{
    if (!sceneFileInfo.exists()) {
        qCCritical(sceneLoaderLog).noquote() << QStringLiteral("File %1 does not exist").arg(sceneFileInfo.fileName());
        return false;
    }
    if (!cacheDir.exists()) {
        qCCritical(sceneLoaderLog).noquote() << QStringLiteral("Dir %1 does not exist").arg(cacheDir.path());
        return false;
    }
    QFileInfo cacheEntryFileInfo = getCacheEntryFileInfo(sceneFileInfo, cacheDir);
    if (cacheEntryFileInfo.exists()) {
        qCInfo(sceneLoaderLog).noquote() << QStringLiteral("triangles file for scene %1 exists").arg(sceneFileInfo.filePath());
        return loadFromCache(cacheEntryFileInfo);
    }
    if (!load(sceneFileInfo)) {
        return false;
    }
    if (!storeToCache(cacheEntryFileInfo)) {
        return false;
    }
    return true;
}
}  // namespace scene_loader

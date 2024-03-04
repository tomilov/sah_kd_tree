#include <scene_data/scene_data.hpp>
#include <scene_loader/assimp_wrappers.hpp>
#include <scene_loader/scene_loader.hpp>
#include <utils/auto_cast.hpp>

#include <assimp/Importer.hpp>
#include <assimp/matrix3x3.h>
#include <assimp/matrix4x4.h>
#include <assimp/postprocess.h>
#include <assimp/quaternion.h>
#include <assimp/scene.h>
#include <assimp/vector3.h>
#include <glm/common.hpp>
#include <glm/ext/quaternion_double.hpp>
#include <glm/ext/quaternion_float.hpp>
#include <glm/ext/vector_double3.hpp>
#include <glm/ext/vector_double4.hpp>
#include <glm/ext/vector_float3.hpp>
#include <glm/ext/vector_float4.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <QtCore/QByteArray>
#include <QtCore/QCryptographicHash>
#include <QtCore/QDataStream>
#include <QtCore/QDebug>
#include <QtCore/QDir>
#include <QtCore/QDirIterator>
#include <QtCore/QElapsedTimer>
#include <QtCore/QFile>
#include <QtCore/QFileInfo>
#include <QtCore/QLocale>
#include <QtCore/QLoggingCategory>
#include <QtCore/QSaveFile>
#include <QtCore/QString>
#include <QtCore/QStringList>

#include <algorithm>
#include <iterator>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace Qt::StringLiterals;

namespace scene_loader
{
namespace
{
Q_DECLARE_LOGGING_CATEGORY(sceneLoaderLog)
Q_LOGGING_CATEGORY(sceneLoaderLog, "scene_loader")

static constexpr qint32 kCurrentCacheFormatVersion = 1;

template<typename Type>
[[nodiscard]] QString toString(const Type & value)
{
    QString string;
    QDebug{&string}.noquote().nospace() << value;
    return string;
}

template<typename T>
[[nodiscard]] QString formattedDataSize(const T & value, int precision = 3)
{
    return QLocale::c().formattedDataSize(utils::autoCast(value), precision);
}

[[nodiscard]] bool checkDataStreamStatus(QDataStream & dataStream, QString description)
{
    auto status = dataStream.status();
    if (status != QDataStream::Ok) {
        qCWarning(sceneLoaderLog).noquote() << u"%1 (data stream error: %2 %3)"_s.arg(description, toString(status), dataStream.device()->errorString());
        return {};
    }
    return true;
}

[[nodiscard]] constexpr glm::vec3 assimpToGlmVector [[maybe_unused]] (const aiVector3D & v)
{
    return glm::vec3{glm::tvec3<ai_real>{v.x, v.y, v.z}};
}

[[nodiscard]] constexpr glm::quat assimpToGlmQuaternion [[maybe_unused]] (const aiQuaternion & q)
{
    return glm::quat{glm::tquat<ai_real>{q.w, q.x, q.y, q.z}};
}

[[nodiscard]] constexpr glm::mat3 assimpToGlmMatrix [[maybe_unused]] (const aiMatrix3x3 & m)
{
    using SrcCol = glm::tvec3<ai_real>;
    using DstCol = glm::mat3::col_type;
    return {
        DstCol{SrcCol{m.a1, m.b1, m.c1}},
        DstCol{SrcCol{m.a2, m.b2, m.c2}},
        DstCol{SrcCol{m.a3, m.b3, m.c3}},
    };
}

[[nodiscard]] constexpr glm::mat4 assimpToGlmMatrix [[maybe_unused]] (const aiMatrix4x4 & m)
{
    using SrcCol = glm::tvec4<ai_real>;
    using DstCol = glm::mat4::col_type;
    return {
        DstCol{SrcCol{m.a1, m.b1, m.c1, m.d1}},
        DstCol{SrcCol{m.a2, m.b2, m.c2, m.d2}},
        DstCol{SrcCol{m.a3, m.b3, m.c3, m.d3}},
        DstCol{SrcCol{m.a4, m.b4, m.c4, m.d4}},
    };
}

[[nodiscard]] QFileInfo getCacheFileInfo(QFileInfo sceneFileInfo, QDir cacheDir)
{
    QFile sceneFile{sceneFileInfo.filePath()};
    if (!sceneFile.open(QFile::ReadOnly)) {
        return {};
    }
    QCryptographicHash cryptographicHash{QCryptographicHash::Algorithm::Blake2s_256};
    cryptographicHash.addData(sceneFileInfo.filePath().toUtf8());
    if (!cryptographicHash.addData(&sceneFile)) {
        return {};
    }
    QFileInfo cacheFileInfo;
    cacheFileInfo.setFile(cacheDir, QString::fromUtf8(cryptographicHash.result().toHex()).append(".triangle"));
    return cacheFileInfo;
}

[[nodiscard]] bool loadFromCache(scene_data::SceneData & scene, QFileInfo cacheFileInfo)
{
    QFile cacheFile{cacheFileInfo.filePath()};
    if (!cacheFile.open(QFile::ReadOnly)) {
        qCCritical(sceneLoaderLog).noquote() << u"unable to open file %1 to read scene from: %2"_s.arg(cacheFile.fileName(), toString(cacheFile.error()));
        return {};
    }

    qCInfo(sceneLoaderLog).noquote() << u"start to load scene from file %1"_s.arg(cacheFile.fileName());
    QElapsedTimer loadTimer;
    loadTimer.start();

    QDataStream dataStream{&cacheFile};

    std::remove_const_t<decltype(kCurrentCacheFormatVersion)> cacheFormatVersion;
    if (!checkDataStreamStatus(dataStream >> cacheFormatVersion, u"unable to read format version from file %1"_s.arg(cacheFile.fileName()))) {
        return {};
    }
    if (cacheFormatVersion != kCurrentCacheFormatVersion) {
        qCWarning(sceneLoaderLog).noquote() << u"format version %1 for scene cache file %2 does not match current format version %3"_s.arg(cacheFormatVersion).arg(cacheFile.fileName()).arg(kCurrentCacheFormatVersion);
        return {};
    }

    const auto loadDataFromCache = [&dataStream, &cacheFile](auto * data, size_t count, QString dataName) -> bool
    {
        static_assert(std::is_standard_layout_v<std::remove_reference_t<decltype(*data)>>, "!");
        auto d = utils::safeCast<char *>(data);
        size_t dataSize = count * sizeof *data;
        qCDebug(sceneLoaderLog).noquote() << u"loadDataFromCache %1\t\t%2"_s.arg(dataSize).arg(dataName);
        while (dataSize > 0) {
            int size = std::numeric_limits<int>::max();
            if (utils::safeCast<size_t>(size) > dataSize) {
                size = utils::safeCast<int>(dataSize);
            }
            int readSize = dataStream.readRawData(d, size);
            if (size != readSize) {
                qCInfo(sceneLoaderLog).noquote() << u"unable to read %1 array from scene cache file %2: need %3 bytes, read %4 bytes"_s.arg(dataName, cacheFile.fileName()).arg(size).arg(readSize);
                return {};
            }
            if (!checkDataStreamStatus(dataStream, u"unable to read %1 array from scene cache file %2"_s.arg(dataName, cacheFile.fileName()))) {
                return {};
            }
            INVARIANT(dataSize >= utils::safeCast<size_t>(size), "{} ^ {}", dataSize, size);
            dataSize -= size;
            d += size;
        }
        return true;
    };
    const auto loadVectorFromCache = [&dataStream, &cacheFile, &loadDataFromCache](auto & vector, QString arrayName) -> bool
    {
        qint32 arrayLength;
        if (!checkDataStreamStatus(dataStream >> arrayLength, u"unable to read size of array of %1 from scene cache file %2"_s.arg(arrayName, cacheFile.fileName()))) {
            return {};
        }
        qCDebug(sceneLoaderLog).noquote() << u"loadVectorFromCache %1\t\t%2"_s.arg(arrayLength).arg(arrayName);
        vector.resize(utils::autoCast(arrayLength));
        if (!loadDataFromCache(std::data(vector), std::size(vector), arrayName)) {
            return {};
        }
        return true;
    };
    const auto loadArrayFromCache = [&dataStream, &cacheFile, &loadDataFromCache]<typename T>(utils::MemArray<T> & array, QString arrayName) -> bool
    {
        qint32 arrayLength;
        if (!checkDataStreamStatus(dataStream >> arrayLength, u"unable to read size of array of %1 from scene cache file %2"_s.arg(arrayName, cacheFile.fileName()))) {
            return {};
        }
        qCDebug(sceneLoaderLog).noquote() << u"loadArrayFromCache %1\t\t%2"_s.arg(arrayLength).arg(arrayName);
        array = utils::MemArray<T>{utils::autoCast(arrayLength)};
        if (!loadDataFromCache(array.begin(), array.getCount(), arrayName)) {
            return {};
        }
        return true;
    };

    quint64 sceneNodeCount;
    if (!checkDataStreamStatus(dataStream >> sceneNodeCount, u"unable to read number of nodes from scene cache file %1"_s.arg(cacheFile.fileName()))) {
        return {};
    }
    qCDebug(sceneLoaderLog).noquote() << u"nodeCount %1"_s.arg(sceneNodeCount);
    scene.nodes.resize(utils::autoCast(sceneNodeCount));
    for (scene_data::Node & node : scene.nodes) {
        if (!loadDataFromCache(&node.transform, 1, "node.transform")) {
            return {};
        }
        if (!loadVectorFromCache(node.meshes, "node.meshes")) {
            return {};
        }
        if (!loadVectorFromCache(node.children, "node.children")) {
            return {};
        }
        if (!loadDataFromCache(&node.aabb, 1, "node.aabb")) {
            return {};
        }
    }
    if (!loadVectorFromCache(scene.meshes, "scene.meshes")) {
        return {};
    }
    if (!loadDataFromCache(&scene.aabb, 1, "scene.aabb")) {
        return {};
    }
    if (!loadArrayFromCache(scene.indices, "scene.indices")) {
        return {};
    }
    if (!loadArrayFromCache(scene.vertices, "scene.vertices")) {
        return {};
    }
    if (!dataStream.atEnd()) {
        qCWarning(sceneLoaderLog).noquote() << u"scene cache file %1 contain extra data at the end"_s.arg(cacheFile.fileName());
    }

    qCInfo(sceneLoaderLog).noquote() << u"scene successfuly loaded from scene cache file %1 (size %2) in %3 ms"_s.arg(cacheFile.fileName(), formattedDataSize(cacheFile.size())).arg(loadTimer.nsecsElapsed() * 1E-6);
    qCDebug(sceneLoaderLog).noquote() << u"scene: %1 meshes, %2 indices, %3 vertices"_s.arg(std::size(scene.meshes)).arg(scene.indices.getCount()).arg(scene.vertices.getCount());
    return true;
}

[[nodiscard]] bool storeToCache(scene_data::SceneData & scene, QFileInfo cacheFileInfo)
{
    QSaveFile cacheFile{cacheFileInfo.filePath()};
    if (!cacheFile.open(QFile::OpenModeFlag::WriteOnly)) {
        qCCritical(sceneLoaderLog).noquote() << u"unable to open file %1 to write scene to: %2"_s.arg(cacheFile.fileName(), toString(cacheFile.error()));
        return {};
    }

    qCInfo(sceneLoaderLog).noquote() << u"start to save scene to file %1"_s.arg(cacheFile.fileName());
    QElapsedTimer saveTimer;
    saveTimer.start();

    QDataStream dataStream{&cacheFile};

    if (!checkDataStreamStatus(dataStream << kCurrentCacheFormatVersion, u"unable to write format version to scene cache file %1"_s.arg(cacheFile.fileName()))) {
        return {};
    }

    const auto saveDataToCache = [&dataStream, &cacheFile](const auto * data, size_t count, QString dataName) -> bool
    {
        static_assert(std::is_standard_layout_v<std::remove_reference_t<decltype(*data)>>, "!");
        auto d = utils::safeCast<const char *>(data);
        size_t dataSize = count * sizeof *data;
        qCDebug(sceneLoaderLog).noquote() << u"saveDataToCache %1\t\t%2"_s.arg(dataSize).arg(dataName);
        while (dataSize > 0) {
            int size = std::numeric_limits<int>::max();
            if (utils::safeCast<size_t>(size) > dataSize) {
                size = utils::safeCast<int>(dataSize);
            }
            int writeSize = dataStream.writeRawData(d, size);
            if (size != writeSize) {
                qCInfo(sceneLoaderLog).noquote() << u"unable to write array %1 to scene cache file %2: want %3 bytes, written %4 bytes"_s.arg(dataName, cacheFile.fileName()).arg(size).arg(writeSize);
                return {};
            }
            if (!checkDataStreamStatus(dataStream, u"unable to write array %1 to scene cache file %2"_s.arg(dataName, cacheFile.fileName()))) {
                return {};
            }
            INVARIANT(dataSize >= utils::safeCast<size_t>(size), "{} ^ {}", dataSize, size);
            dataSize -= size;
            d += size;
        }
        return true;
    };
    const auto saveVectorToCache = [&dataStream, &cacheFile, &saveDataToCache](const auto & vector, QString arrayName) -> bool
    {
        qint32 arrayLength = utils::autoCast(std::size(vector));
        qCDebug(sceneLoaderLog).noquote() << u"saveVectorToCache %1\t\t%2"_s.arg(arrayLength).arg(arrayName);
        if (!checkDataStreamStatus(dataStream << arrayLength, u"unable to write size of array of %1 to scene cache file %2"_s.arg(arrayName, cacheFile.fileName()))) {
            return {};
        }
        if (!saveDataToCache(std::data(vector), std::size(vector), arrayName)) {
            return {};
        }
        return true;
    };
    const auto saveArrayToCache = [&dataStream, &cacheFile, &saveDataToCache]<typename T>(const utils::MemArray<T> & array, QString arrayName) -> bool
    {
        qint32 arrayLength = utils::autoCast(array.getCount());
        qCDebug(sceneLoaderLog).noquote() << u"saveArrayToCache %1\t\t%2"_s.arg(arrayLength).arg(arrayName);
        if (!checkDataStreamStatus(dataStream << arrayLength, u"unable to write size of array of %1 to scene cache file %2"_s.arg(arrayName, cacheFile.fileName()))) {
            return {};
        }
        if (!saveDataToCache(array.begin(), array.getCount(), arrayName)) {
            return {};
        }
        return true;
    };

    quint64 sceneNodeCount = utils::autoCast(std::size(scene.nodes));
    qCDebug(sceneLoaderLog).noquote() << u"nodeCount %1"_s.arg(sceneNodeCount);
    if (!checkDataStreamStatus(dataStream << sceneNodeCount, u"unable to write number of nodes to scene cache file %1"_s.arg(cacheFile.fileName()))) {
        return {};
    }
    for (const scene_data::Node & node : scene.nodes) {
        if (!saveDataToCache(&node.transform, 1, "node.transform")) {
            return {};
        }
        if (!saveVectorToCache(node.meshes, "node.meshes")) {
            return {};
        }
        if (!saveVectorToCache(node.children, "node.children")) {
            return {};
        }
        if (!saveDataToCache(&node.aabb, 1, "node.aabb")) {
            return {};
        }
    }
    if (!saveVectorToCache(scene.meshes, "scene.meshes")) {
        return {};
    }
    if (!saveDataToCache(&scene.aabb, 1, "scene.aabb")) {
        return {};
    }
    if (!saveArrayToCache(scene.indices, "scene.indices")) {
        return {};
    }
    if (!saveArrayToCache(scene.vertices, "scene.vertices")) {
        return {};
    }

    if (!cacheFile.commit()) {
        qCWarning(sceneLoaderLog).noquote() << u"failed to commit scene cache to file '%1': %2"_s.arg(cacheFile.fileName(), cacheFile.errorString());
        return {};
    }

    qCInfo(sceneLoaderLog).noquote() << u"scene successfuly saved to scene cache file %1 (size %2) in %3 ms"_s.arg(cacheFile.fileName(), formattedDataSize(cacheFile.size())).arg(saveTimer.nsecsElapsed() * 1E-6);
    qCDebug(sceneLoaderLog).noquote() << u"scene: %1 meshes, %2 indices, %3 vertices"_s.arg(std::size(scene.meshes)).arg(scene.indices.getCount()).arg(scene.vertices.getCount());
    return true;
}
}  // namespace

QStringList getSupportedExtensions()
{
    aiString extensionsString;
    Assimp::Importer{}.GetExtensionList(extensionsString);
    return QString::fromUtf8(QByteArray{extensionsString.data, utils::autoCast(extensionsString.length)}).split(u';');
}

bool load(scene_data::SceneData & scene, QFileInfo sceneFileInfo)
{
    INVARIANT(sceneFileInfo.isFile(), "");

    AssimpLoggerGuard loggerGuard{Assimp::Logger::LogSeverity::VERBOSE};
    Assimp::Importer importer;
    importer.SetProgressHandler(new AssimpProgressHandler);
    importer.SetIOHandler(new AssimpIOSystem);

    auto pFlags =  // aiProcess_CalcTangentSpace | aiProcess_GenSmoothNormals | aiProcess_GenUVCoords |
        aiProcess_JoinIdenticalVertices | aiProcess_ImproveCacheLocality | aiProcess_SplitLargeMeshes | aiProcess_Triangulate | aiProcess_SortByPType | aiProcess_FindInstances | aiProcess_ValidateDataStructure | aiProcess_OptimizeMeshes
        | aiProcess_OptimizeGraph | aiProcess_RemoveComponent;
    // pFlags |= aiProcess_TransformUVCoords;
    if ((pFlags & aiProcess_FindDegenerates) != 0) {  // omit aiProcess_FindDegenerates for tests
        importer.SetPropertyBool(AI_CONFIG_PP_FD_REMOVE, true);
        importer.SetPropertyBool(AI_CONFIG_PP_FD_CHECKAREA, false);  // required for e.g. buddha.obj
    }
    if ((pFlags & aiProcess_SortByPType) != 0) {
        importer.SetPropertyInteger(AI_CONFIG_PP_SBP_REMOVE, aiPrimitiveType_POINT | aiPrimitiveType_LINE);
    }
    if ((pFlags & aiProcess_RemoveComponent) != 0) {
        // all except aiComponent_MESHES
        constexpr auto excludeComponents = aiComponent_NORMALS | aiComponent_TANGENTS_AND_BITANGENTS | aiComponent_COLORS | aiComponent_TEXCOORDS | aiComponent_BONEWEIGHTS | aiComponent_ANIMATIONS | aiComponent_TEXTURES | aiComponent_LIGHTS
                                           | aiComponent_CAMERAS | aiComponent_MATERIALS;
        importer.SetPropertyInteger(AI_CONFIG_PP_RVC_FLAGS, excludeComponents);
    }
    INVARIANT(importer.ValidateFlags(pFlags), "");

    {
        QElapsedTimer sceneLoadTimer;
        sceneLoadTimer.start();
        if (!importer.ReadFile(qPrintable(QDir::toNativeSeparators(sceneFileInfo.filePath())), pFlags)) {
            qCCritical(sceneLoaderLog).noquote() << u"unable to load scene %1: %2"_s.arg(sceneFileInfo.filePath(), QString::fromUtf8(importer.GetErrorString()));
            return {};
        }
        qCDebug(sceneLoaderLog).noquote() << u"scene loaded in %1 ms"_s.arg(sceneLoadTimer.nsecsElapsed() * 1E-6);
    }

    {
        aiMemoryInfo memoryInfo;
        importer.GetMemoryRequirements(memoryInfo);
        qCDebug(sceneLoaderLog).noquote() << u"scene memory info (%1 total): textures %2, materials %3, meshes %4, nodes %5, animations %6, cameras %7, lights %8"_s.arg(memoryInfo.total)
                                                 .arg(memoryInfo.textures)
                                                 .arg(memoryInfo.materials)
                                                 .arg(memoryInfo.meshes)
                                                 .arg(memoryInfo.nodes)
                                                 .arg(memoryInfo.animations)
                                                 .arg(memoryInfo.cameras)
                                                 .arg(memoryInfo.lights);
    }

    auto assimpScene = importer.GetScene();

    qCInfo(sceneLoaderLog) << "scene has animations:" << assimpScene->HasAnimations();
    qCInfo(sceneLoaderLog) << "scene has cameras:" << assimpScene->HasCameras();
    qCInfo(sceneLoaderLog) << "scene has lights:" << assimpScene->HasLights();
    qCInfo(sceneLoaderLog) << "scene has materials (required if not flag set):" << assimpScene->HasMaterials();
    qCInfo(sceneLoaderLog) << "scene has meshes (required if not flag set):" << assimpScene->HasMeshes();
    qCInfo(sceneLoaderLog) << "scene has textures:" << assimpScene->HasTextures();
    if (((importer.GetPropertyInteger(AI_CONFIG_PP_RVC_FLAGS) & aiComponent_MESHES) == 0) && !assimpScene->HasMeshes()) {
        qCCritical(sceneLoaderLog).noquote() << u"scene %1 has no meshes"_s.arg(sceneFileInfo.filePath());
        return {};
    }
    if (((importer.GetPropertyInteger(AI_CONFIG_PP_RVC_FLAGS) & aiComponent_MATERIALS) == 0) && !assimpScene->HasMaterials()) {
        qCCritical(sceneLoaderLog).noquote() << u"scene %1 has no materials"_s.arg(sceneFileInfo.filePath());
        return {};
    }

    qCInfo(sceneLoaderLog).noquote() << u"scene flags: %1"_s.arg(assimpScene->mFlags);
    qCInfo(sceneLoaderLog).noquote() << u"number of animations: %1"_s.arg(assimpScene->mNumAnimations);
    qCInfo(sceneLoaderLog).noquote() << u"number of cameras: %1"_s.arg(assimpScene->mNumCameras);
    qCInfo(sceneLoaderLog).noquote() << u"number of lights: %1"_s.arg(assimpScene->mNumLights);
    qCInfo(sceneLoaderLog).noquote() << u"number of materials: %1"_s.arg(assimpScene->mNumMaterials);
    qCInfo(sceneLoaderLog).noquote() << u"number of meshes: %1"_s.arg(assimpScene->mNumMeshes);
    qCInfo(sceneLoaderLog).noquote() << u"number of textures: %1"_s.arg(assimpScene->mNumTextures);

    auto assimpRootNode = assimpScene->mRootNode;
    if (!assimpRootNode) {
        qCCritical(sceneLoaderLog).noquote() << u"scene %1 has no root node"_s.arg(sceneFileInfo.filePath());
        return {};
    }

    struct MeshUsage
    {
        size_t meshIndex;
        size_t useCount = 0;
    };
    std::unordered_map<const aiMesh *, MeshUsage> meshUsages;
    {
        size_t assimpMeshCount = utils::autoCast(assimpScene->mNumMeshes);
        auto assimpMeshes = assimpScene->mMeshes;
        meshUsages.reserve(assimpMeshCount);
        for (size_t assimpMeshIndex = 0; assimpMeshIndex < assimpMeshCount; ++assimpMeshIndex) {
            auto assimpMesh = assimpMeshes[assimpMeshIndex];
            MeshUsage meshUsage = {
                .meshIndex = std::size(meshUsages),
            };
            if (const auto & [m, inserted] = meshUsages.emplace(assimpMesh, meshUsage); !inserted) {
                INVARIANT(false, "Duplicated mesh #{}: #{}", m->second.meshIndex, meshUsage.meshIndex);
            }
        }

        std::unordered_map<const aiNode *, size_t> parents;
        const auto traverseNodes = [&scene, &parents, &meshUsages, assimpMeshes](const auto & traverseNodes, const aiNode * assimpNode) -> size_t
        {
            size_t nodeIndex = std::size(scene.nodes);
            scene_data::Node & node = scene.nodes.emplace_back();
            if (auto assimpNodeParent = assimpNode->mParent) {
                auto p = parents.find(assimpNodeParent);
                ASSERT(p != std::end(parents));
                node.parent = p->second;
            } else {
                node.parent = nodeIndex;
            }
            parents.emplace(assimpNode, nodeIndex);
            node.transform = assimpToGlmMatrix(assimpNode->mTransformation);
            if (assimpNode->mNumMeshes > 0) {
                size_t assimpMeshCount = utils::autoCast(assimpNode->mNumMeshes);
                node.meshes.reserve(assimpMeshCount);
                for (size_t m = 0; m < assimpMeshCount; ++m) {
                    auto assimpMeshIndex = assimpNode->mMeshes[m];
                    auto assimpMesh = assimpMeshes[assimpMeshIndex];
                    auto u = meshUsages.find(assimpMesh);
                    ASSERT(u != std::end(meshUsages));
                    MeshUsage & meshUsage = u->second;
                    node.meshes.push_back(meshUsage.meshIndex);
                    ++meshUsage.useCount;
                }
            }
            ASSERT(std::empty(node.children));
            size_t childrenCount = utils::autoCast(assimpNode->mNumChildren);
            node.children.reserve(childrenCount);
            auto assimpNodeChildren = assimpNode->mChildren;
            for (size_t c = 0; c < childrenCount; ++c) {
                auto assimpNodeChild = assimpNodeChildren[c];
                ASSERT(assimpNodeChild->mParent == assimpNode);
                size_t childNodeIndex = traverseNodes(traverseNodes, assimpNodeChild);
                scene.nodes.at(nodeIndex).children.push_back(childNodeIndex);
            }
            return nodeIndex;
        };
        if (traverseNodes(traverseNodes, assimpRootNode) != 0) {
            ASSERT(false);
        }
    }

    using UsedMesh = std::pair<const aiMesh *, MeshUsage>;
    std::vector<UsedMesh> usedMeshes{std::cbegin(meshUsages), std::cend(meshUsages)};
    uint32_t indexCount = 0;
    uint32_t vertexCount = 0;
    {
        constexpr auto isMeshUsed = [](const UsedMesh & usedMesh)
        {
            return usedMesh.second.useCount == 0;
        };
        auto u = std::remove_if(std::begin(usedMeshes), std::end(usedMeshes), isMeshUsed);
        usedMeshes.erase(u, std::end(usedMeshes));

        qCInfo(sceneLoaderLog).noquote() << u"number of meshes in assimp scene: %1"_s.arg(assimpScene->mNumMeshes);
        qCInfo(sceneLoaderLog).noquote() << u"expected number of meshes: %1"_s.arg(std::size(meshUsages));
        qCInfo(sceneLoaderLog).noquote() << u"actual number of meshes used: %1"_s.arg(std::size(usedMeshes));

        constexpr auto isMeshUsedSooner = [](const UsedMesh & l, const UsedMesh & r) -> bool
        {
            return l.second.meshIndex < r.second.meshIndex;
        };
        std::sort(std::begin(usedMeshes), std::end(usedMeshes), isMeshUsedSooner);

        std::unordered_map<size_t, size_t> meshIndexRemap;
        meshIndexRemap.reserve(std::size(usedMeshes));
        for (auto & [assimpMesh, meshUsage] : usedMeshes) {
            if ((assimpMesh->mPrimitiveTypes & ~(aiPrimitiveType_TRIANGLE | aiPrimitiveType_NGONEncodingFlag)) != 0) {
                qCWarning(sceneLoaderLog).noquote() << u"primitive type of mesh %1 is not triangle (possibly NGON-encoded)"_s.arg(QString::fromLatin1(assimpMesh->mName.C_Str()));
                return {};
            }
            size_t newMeshIndex = std::size(scene.meshes);
            meshIndexRemap.emplace(meshUsage.meshIndex, newMeshIndex);
            meshUsage.meshIndex = newMeshIndex;
            auto & mesh = scene.meshes.emplace_back();
            mesh = {
                .indexOffset = indexCount,
                .indexCount = utils::autoCast(assimpMesh->mNumFaces * 3),
                .vertexOffset = vertexCount,
                .vertexCount = utils::autoCast(assimpMesh->mNumVertices),
            };
            indexCount += mesh.indexCount;
            vertexCount += mesh.vertexCount;
        }

        for (scene_data::Node & node : scene.nodes) {
            for (size_t & meshIndex : node.meshes) {
                meshIndex = meshIndexRemap.at(meshIndex);
            }
        }
    }
    qCInfo(sceneLoaderLog).noquote() << u"total number of faces: %1"_s.arg(indexCount / 3);
    qCInfo(sceneLoaderLog).noquote() << u"total number of vertices: %1"_s.arg(vertexCount);

    {
        scene.indices = utils::MemArray<uint32_t>{indexCount};
        scene.vertices = utils::MemArray<scene_data::VertexAttributes>{vertexCount};
        for (const auto & [assimpMesh, meshUsage] : usedMeshes) {
            const auto & mesh = scene.meshes.at(meshUsage.meshIndex);
            auto & aabb = scene.meshes.at(meshUsage.meshIndex).aabb;

            {
                auto vertex = std::next(scene.vertices.begin(), mesh.vertexOffset);
                const auto vertexEnd = std::next(vertex, mesh.vertexCount);
                auto assimpVertices = assimpMesh->mVertices;
                for (uint32_t v = 0; v < mesh.vertexCount; ++v) {
                    auto & vertexAttributes = *vertex++;
                    vertexAttributes.position = assimpToGlmVector(assimpVertices[v]);
                    aabb.min = glm::min(aabb.min, vertexAttributes.position);
                    aabb.max = glm::max(aabb.max, vertexAttributes.position);
                    // another vertex attributes: mNormals, mTangents, mBitangents, mColors, mTextureCoords+mNumUVComponents
                }
                ASSERT(vertex == vertexEnd);
            }

            auto sceneIndices = scene.indices.begin();

            if (mesh.indexCount == 0) {
                INVARIANT((mesh.vertexCount % 3) == 0, "Vertex count {} is not multiple of 3 in mesh {}", mesh.vertexCount, meshUsage.meshIndex);
                continue;
            }

            {
                ASSERT_MSG((mesh.indexOffset % 3) == 0, "{} {}", mesh.indexOffset % 3, mesh.indexOffset);
                ASSERT_MSG((mesh.indexCount % 3) == 0, "{} {}", mesh.indexCount % 3, mesh.indexCount);
                auto index = std::next(sceneIndices, mesh.indexOffset);
                const auto indexEnd = std::next(index, mesh.indexCount);
                auto assimpFaces = assimpMesh->mFaces;
                for (uint32_t f = 0; f < mesh.indexCount / 3; ++f) {
                    const aiFace & assimpFace = assimpFaces[f];
                    INVARIANT(assimpFace.mNumIndices == 3, "{}", assimpFace.mNumIndices);
                    auto assimpFaceIndices = assimpFace.mIndices;
                    *index++ = utils::autoCast(assimpFaceIndices[0]);
                    *index++ = utils::autoCast(assimpFaceIndices[1]);
                    *index++ = utils::autoCast(assimpFaceIndices[2]);
                }
                ASSERT(index == indexEnd);
            }

            if ((true)) {
                std::vector<uint32_t> vertexUseCounts(mesh.vertexCount);
                auto index = std::next(scene.indices.begin(), mesh.indexOffset);
                const auto indexEnd = std::next(index, mesh.indexCount);
                for (uint32_t i : std::span<const uint32_t>(index, indexEnd)) {
                    ++vertexUseCounts.at(i);
                }
                for (uint32_t vertexUseCount : vertexUseCounts) {
                    if (vertexUseCount == 0) {
                        qCWarning(sceneLoaderLog).noquote() << u"Vertex %1 is not used in mesh %2"_s.arg(vertexUseCount).arg(meshUsage.meshIndex);
                    }
                }
            }
        }
    }

    scene.updateAABBs();

    importer.FreeScene();

    return true;
}

bool cachingLoad(scene_data::SceneData & scene, QFileInfo sceneFileInfo, QDir cacheDir)
{
    if ((true)) {
        QStringList nameFilters;
        nameFilters << "*.triangle";
        QDirIterator cachedScenes{cacheDir.path(), nameFilters, QDir::Filter::Files};
        qint64 size = 0;
        while (cachedScenes.hasNext()) {
            auto fileInfo = cachedScenes.nextFileInfo();
            size += fileInfo.size();
        }
        qCDebug(sceneLoaderLog).noquote() << u"Cache size %1"_s.arg(formattedDataSize(size));
    }
    if (!sceneFileInfo.exists()) {
        qCCritical(sceneLoaderLog).noquote() << u"file %1 does not exist"_s.arg(sceneFileInfo.fileName());
        return {};
    }
    if (!cacheDir.exists()) {
        qCCritical(sceneLoaderLog).noquote() << u"dir %1 does not exist"_s.arg(cacheDir.path());
        return {};
    }
    QFileInfo cacheFileInfo = getCacheFileInfo(sceneFileInfo, cacheDir);
    if (cacheFileInfo.exists()) {
        qCInfo(sceneLoaderLog).noquote() << u"scene file for scene %1 exists"_s.arg(sceneFileInfo.filePath());
        if (loadFromCache(scene, cacheFileInfo)) {
            return true;
        }
        if (!QFile::remove(cacheFileInfo.filePath())) {
            qCWarning(sceneLoaderLog).noquote() << u"unable to remove broken cache file %1"_s.arg(cacheFileInfo.filePath());
        }
        qCWarning(sceneLoaderLog).noquote() << u"cache broken or cannot be read; scene will be loaded from file %1"_s.arg(sceneFileInfo.filePath());
    }
    if (!load(scene, sceneFileInfo)) {
        return {};
    }
    if (!storeToCache(scene, cacheFileInfo)) {
        return {};
    }
    return true;
}
}  // namespace scene_loader

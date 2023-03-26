#include <scene/scene.hpp>
#include <scene_loader/assimp_wrappers.hpp>
#include <scene_loader/scene_loader.hpp>
#include <utils/auto_cast.hpp>

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>
#include <glm/gtc/type_ptr.hpp>

#include <QtCore/QCryptographicHash>
#include <QtCore/QDataStream>
#include <QtCore/QDebug>
#include <QtCore/QDir>
#include <QtCore/QElapsedTimer>
#include <QtCore/QFile>
#include <QtCore/QFileInfo>
#include <QtCore/QLoggingCategory>
#include <QtCore/QSaveFile>
#include <QtCore/QString>
#include <QtCore/QStringList>
#include <QtCore/QByteArray>

using namespace Qt::StringLiterals;

namespace scene_loader
{
namespace
{
Q_DECLARE_LOGGING_CATEGORY(sceneLoaderLog)
Q_LOGGING_CATEGORY(sceneLoaderLog, "scene_loader")

static constexpr qint32 kCurrentCacheFormatVersion = 1;

template<typename Type>
QString toString(const Type & value)
{
    QString string;
    QDebug{&string}.noquote().nospace() << value;
    return string;
}

bool checkDataStreamStatus(QDataStream & dataStream, QString description)
{
    auto status = dataStream.status();
    if (status != QDataStream::Ok) {
        qCWarning(sceneLoaderLog).noquote() << u"%1 (data stream error: %2 %3)"_s.arg(description, toString(status), dataStream.device()->errorString());
        return {};
    }
    return true;
}
}  // namespace

bool SceneLoader::load(scene::Scene & scene, QFileInfo sceneFileInfo) const
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

    uint32_t indexCount = 0;
    uint32_t vertexCount = 0;
    std::unordered_map<const aiMesh *, uint32_t> instances;
    {
        size_t meshCount = utils::autoCast(assimpScene->mNumMeshes);
        scene.meshes.resize(meshCount);
        qCInfo(sceneLoaderLog).noquote() << u"total number of meshes: %1"_s.arg(std::size(scene.meshes));
        const auto & assimpMeshes = assimpScene->mMeshes;
        for (uint32_t m = 0; m < meshCount; ++m) {
            auto assimpMesh = assimpMeshes[m];
            if ((assimpMesh->mPrimitiveTypes & ~(aiPrimitiveType_TRIANGLE | aiPrimitiveType_NGONEncodingFlag)) != 0) {
                qCWarning(sceneLoaderLog).noquote() << u"primitive type of mesh %1 is not triangle (possibly NGON-encoded)"_s.arg(m);
                return {};
            }
            // qCDebug(sceneLoaderLog).noquote() << assimpMeshPointer->mName.C_Str();
            auto & mesh = scene.meshes.at(m);
            if (auto [instance, inserted] = instances.emplace(assimpMesh, m); inserted) {
                mesh = {
                    .indexOffset = indexCount,
                    .indexCount = utils::autoCast(assimpMesh->mNumFaces * 3),
                    .vertexOffset = vertexCount,
                    .vertexCount = utils::autoCast(assimpMesh->mNumVertices),
                };
                indexCount += mesh.indexCount;
                vertexCount += mesh.vertexCount;
            } else {
                INVARIANT(instance->second < m, "");
                mesh = scene.meshes.at(instance->second);
            }
        }
    }
    qCInfo(sceneLoaderLog).noquote() << u"total number of instances: %1"_s.arg(std::size(instances));
    qCInfo(sceneLoaderLog).noquote() << u"total number of vertices: %1"_s.arg(vertexCount);
    qCInfo(sceneLoaderLog).noquote() << u"total number of faces: %1"_s.arg(indexCount);

    {
        scene.resizeIndices(indexCount);
        scene.resizeVertices(vertexCount);
        static constexpr auto toVertex = [](const aiVector3D & v) -> glm::vec3
        {
            if constexpr (std::is_same_v<ai_real, double>) {
                return utils::autoCast(glm::dvec3{v.x, v.y, v.z});
            } else {
                static_assert(std::is_same_v<ai_real, float>);
                return {v.x, v.y, v.z};
            }
        };
        for (const auto & [assimpMeshPointer, m] : instances) {
            auto [indexOffset, indexCount, vertexOffset, vertexCount] = scene.meshes[m];
            INVARIANT((indexCount % 3) == 0, "");
            const aiMesh & assimpMesh = *assimpMeshPointer;
            auto index = std::next(scene.indices.get(), indexOffset);
            const auto indexEnd = std::next(index, indexCount);
            for (uint32_t f = 0; f < indexCount / 3; ++f) {
                const aiFace & assimpFace = assimpMesh.mFaces[f];
                INVARIANT(assimpFace.mNumIndices == 3, "");
                *index++ = utils::autoCast(assimpFace.mIndices[0]);
                *index++ = utils::autoCast(assimpFace.mIndices[1]);
                *index++ = utils::autoCast(assimpFace.mIndices[2]);
            }
            INVARIANT(index == indexEnd, "");
            auto vertex = std::next(scene.vertices.get(), vertexOffset);
            const auto vertexEnd = std::next(vertex, vertexCount);
            for (uint32_t v = 0; v < vertexCount; ++v) {
                auto & vertexAttributes = *vertex++;
                vertexAttributes.position = toVertex(assimpMesh.mVertices[v]);
                // another vertex attributes
            }
            INVARIANT(vertex == vertexEnd, "");
        }
    }

    {
        std::unordered_map<const aiNode *, size_t> parents;
        static constexpr auto toMatrix = [](const aiMatrix4x4 & m) -> glm::mat4
        {
            return {
                m.a1, m.a2, m.a3, m.a4, m.b1, m.b2, m.b3, m.b4, m.c1, m.c2, m.c3, m.c4, m.d1, m.d2, m.d3, m.d4,
            };
        };
        const auto traverseNodes = [&scene, &parents](const auto & traverseNodes, const aiNode & assimpNode) -> size_t
        {
            size_t nodeIndex = scene.nodes.size();
            scene::Node & node = scene.nodes.emplace_back();
            if (assimpNode.mParent) {
                INVARIANT(parents.find(assimpNode.mParent) != std::end(parents), "");
                node.parent = parents.find(assimpNode.mParent)->second;
            } else {
                node.parent = nodeIndex;
            }
            parents.emplace(&assimpNode, nodeIndex);
            node.transform = toMatrix(assimpNode.mTransformation);
            if (assimpNode.mNumMeshes > 0) {
                size_t meshCount = utils::autoCast(assimpNode.mNumMeshes);
                node.meshes.resize(meshCount);
                for (size_t m = 0; m < meshCount; ++m) {
                    node.meshes[m] = utils::autoCast(assimpNode.mMeshes[m]);
                }
            }
            INVARIANT(std::empty(node.children), "");
            node.children.reserve(utils::autoCast(assimpNode.mNumChildren));
            for (unsigned int c = 0; c < assimpNode.mNumChildren; ++c) {
                size_t child = traverseNodes(traverseNodes, *assimpNode.mChildren[c]);
                scene.nodes[nodeIndex].children.push_back(child);
            }
            return nodeIndex;
        };
        if (traverseNodes(traverseNodes, *assimpRootNode) != 0) {
            INVARIANT(false, "");
        }
    }
    importer.FreeScene();

    return true;
}

QFileInfo SceneLoader::getCacheFileInfo(QFileInfo sceneFileInfo, QDir cacheDir) const
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
    QFileInfo cacheFileInfo;
    cacheFileInfo.setFile(cacheDir, QString::fromUtf8(cryptographicHash.result().toHex()).append(".triangle"));
    return cacheFileInfo;
}

bool SceneLoader::loadFromCache(scene::Scene & scene, QFileInfo cacheFileInfo) const
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
        int size = utils::autoCast(count * sizeof *data);
        int readSize = dataStream.readRawData(utils::autoCast(data), size);
        if (size != readSize) {
            qCInfo(sceneLoaderLog).noquote() << u"unable to read %1 array from scene cache file %2: need %3 bytes, read %4 bytes"_s.arg(dataName, cacheFile.fileName()).arg(size).arg(readSize);
            return {};
        }
        if (!checkDataStreamStatus(dataStream, u"unable to read %1 array from scene cache file %2"_s.arg(dataName, cacheFile.fileName()))) {
            return {};
        }
        return true;
    };
    const auto loadVectorFromCache = [&dataStream, &cacheFile, &loadDataFromCache](auto & vector, QString arrayName) -> bool
    {
        qint32 arrayLength;
        if (!checkDataStreamStatus(dataStream >> arrayLength, u"unable to read size of array of %1 from scene cache file %2"_s.arg(arrayName, cacheFile.fileName()))) {
            return {};
        }
        vector.resize(utils::autoCast(arrayLength));
        if (!loadDataFromCache(std::data(vector), std::size(vector), arrayName)) {
            return {};
        }
        return true;
    };
    const auto loadArrayFromCache = [&dataStream, &cacheFile, &loadDataFromCache]<typename T>(std::unique_ptr<T[]> & array, size_t & arraySize, QString arrayName) -> bool
    {
        qint32 arrayLength;
        if (!checkDataStreamStatus(dataStream >> arrayLength, u"unable to read size of array of %1 from scene cache file %2"_s.arg(arrayName, cacheFile.fileName()))) {
            return {};
        }
        arraySize = utils::autoCast(arrayLength);
        array = std::make_unique<T[]>(arraySize);
        if (!loadDataFromCache(array.get(), arraySize, arrayName)) {
            return {};
        }
        return true;
    };

    quint64 sceneNodeCount;
    if (!checkDataStreamStatus(dataStream >> sceneNodeCount, u"unable to read number of nodes from scene cache file %1"_s.arg(cacheFile.fileName()))) {
        return {};
    }
    scene.nodes.resize(utils::autoCast(sceneNodeCount));
    for (scene::Node & node : scene.nodes) {
        float transform[4 * 4];
        if (!loadDataFromCache(&transform, sizeof transform, "node.transform")) {
            return {};
        }
        node.transform = glm::make_mat4x4(transform);
        if (!loadVectorFromCache(node.meshes, "node.meshes")) {
            return {};
        }
        if (!loadVectorFromCache(node.children, "node.children")) {
            return {};
        }
    }
    if (!loadVectorFromCache(scene.meshes, "meshes")) {
        return {};
    }
    if (!loadArrayFromCache(scene.indices, scene.indexCount, "indices")) {
        return {};
    }
    if (!loadArrayFromCache(scene.vertices, scene.vertexCount, "vertices")) {
        return {};
    }
    if (!dataStream.atEnd()) {
        qCWarning(sceneLoaderLog).noquote() << u"scene cache file %1 contain extra data at the end"_s.arg(cacheFile.fileName());
    }

    qCInfo(sceneLoaderLog).noquote() << u"scene successfuly loaded from scene cache file %1 in %2 ms"_s.arg(cacheFile.fileName()).arg(loadTimer.nsecsElapsed() * 1E-6);
    qCDebug(sceneLoaderLog).noquote() << u"scene: %1 meshes, %2 indices, %3 vertices"_s.arg(std::size(scene.meshes)).arg(scene.indexCount).arg(scene.vertexCount);
    return true;
}

bool SceneLoader::storeToCache(scene::Scene & scene, QFileInfo cacheFileInfo) const
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
        int size = int(utils::autoCast(count * sizeof *data));
        int writeSize = dataStream.writeRawData(utils::autoCast(data), size);
        if (size != writeSize) {
            qCInfo(sceneLoaderLog).noquote() << u"unable to write array %1 to scene cache file %2: want %3 bytes, written %4 bytes"_s.arg(dataName, cacheFile.fileName()).arg(size).arg(writeSize);
            return {};
        }
        if (!checkDataStreamStatus(dataStream, u"unable to write array %1 to scene cache file %2"_s.arg(dataName, cacheFile.fileName()))) {
            return {};
        }
        return true;
    };
    const auto saveVectorToCache = [&dataStream, &cacheFile, &saveDataToCache](const auto & vector, QString arrayName) -> bool
    {
        qint32 arrayLength = utils::autoCast(std::size(vector));
        if (!checkDataStreamStatus(dataStream << arrayLength, u"unable to write size of array of %1 to scene cache file %2"_s.arg(arrayName, cacheFile.fileName()))) {
            return {};
        }
        if (!saveDataToCache(std::data(vector), std::size(vector), arrayName)) {
            return {};
        }
        return true;
    };
    const auto saveArrayToCache = [&dataStream, &cacheFile, &saveDataToCache](const auto & array, size_t arraySize, QString arrayName) -> bool
    {
        qint32 arrayLength = utils::autoCast(arraySize);
        if (!checkDataStreamStatus(dataStream << arrayLength, u"unable to write size of array of %1 to scene cache file %2"_s.arg(arrayName, cacheFile.fileName()))) {
            return {};
        }
        if (!saveDataToCache(array.get(), arraySize, arrayName)) {
            return {};
        }
        return true;
    };

    if (!checkDataStreamStatus(dataStream << quint64(utils::autoCast(std::size(scene.nodes))), u"unable to write number of nodes to scene cache file %1"_s.arg(cacheFile.fileName()))) {
        return {};
    }
    for (const scene::Node & node : scene.nodes) {
        if (!saveDataToCache(glm::value_ptr(node.transform), sizeof(float) * 4 * 4, "node.transform")) {
            return {};
        }
        if (!saveVectorToCache(node.meshes, "node.meshes")) {
            return {};
        }
        if (!saveVectorToCache(node.children, "node.children")) {
            return {};
        }
    }
    if (!saveVectorToCache(scene.meshes, "scene.meshes")) {
        return {};
    }
    if (!saveArrayToCache(scene.indices, scene.indexCount, "scene.indices")) {
        return {};
    }
    if (!saveArrayToCache(scene.vertices, scene.vertexCount, "scene.vertices")) {
        return {};
    }

    if (!cacheFile.commit()) {
        qCWarning(sceneLoaderLog).noquote() << u"failed to commit scene cache to file '%1': %2"_s.arg(cacheFile.fileName(), cacheFile.errorString());
        return {};
    }

    qCInfo(sceneLoaderLog).noquote() << u"scene successfuly saved to scene cache file %1 in %2 ms"_s.arg(cacheFile.fileName()).arg(saveTimer.nsecsElapsed() * 1E-6);
    qCDebug(sceneLoaderLog).noquote() << u"scene: %1 meshes, %2 indices, %3 vertices"_s.arg(std::size(scene.meshes)).arg(scene.indexCount).arg(scene.vertexCount);
    return true;
}

bool SceneLoader::cachingLoad(scene::Scene & scene, QFileInfo sceneFileInfo, QDir cacheDir) const
{
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

QStringList SceneLoader::getSupportedExtensions()
{
    aiString extensionsString;
    Assimp::Importer{}.GetExtensionList(extensionsString);
    std::vector<std::string> extensions;
    return QString::fromUtf8(QByteArray{extensionsString.data, utils::autoCast(extensionsString.length)}).split(u';');
}
}  // namespace scene_loader

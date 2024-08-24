#include <builder/builder.hpp>
#include <scene_data/scene_data.hpp>
#include <scene_loader/scene_loader.hpp>
#include <utils/auto_cast.hpp>

#include <gtest/gtest.h>

#include <QtCore/QDebug>
#include <QtCore/QDir>
#include <QtCore/QFileInfo>
#include <QtCore/QLoggingCategory>
#include <QtCore/QStandardPaths>
#include <QtCore/QString>
#include <QtCore/QtLogging>

#include <chrono>
#include <memory>
#include <optional>
#include <ostream>
#include <utility>

#include <cstddef>

#include <cuda_runtime.h>

using namespace Qt::StringLiterals;

namespace
{
Q_DECLARE_LOGGING_CATEGORY(builderTest)
Q_LOGGING_CATEGORY(builderTest, "builder.test")

constexpr float kEmptinessFactor = 0.8f;
constexpr float kTraversalCost = 2.0f;
constexpr float kIntersectionCost = 1.0f;
constexpr int kMaxdepth = 1000;
}  // namespace

class Builder : public testing::Test
{
protected:
    [[nodiscard]] bool buildSceneFromFile(QString sceneFileName, float emptinessFactor = kEmptinessFactor, float traversalCost = kTraversalCost, float intersectionCost = kIntersectionCost, int maxDepth = kMaxdepth) const
    {
        scene_data::SceneData sceneData;
        QFileInfo sceneFileInfo{sceneFileName};
        if ((true)) {
            if (!scene_loader::load(sceneData, sceneFileInfo)) {
                qCDebug(builderTest).noquote() << u"Cannot load scene from file %1"_s.arg(sceneFileName);
                return false;
            }
        } else {
            auto cachePath = QStandardPaths::writableLocation(QStandardPaths::CacheLocation);
            if (!scene_loader::cachingLoad(sceneData, sceneFileInfo, cachePath.isEmpty() ? QDir::temp() : cachePath)) {
                qCDebug(builderTest).noquote() << u"Cannot load scene from file %1"_s.arg(sceneFileName);
                return false;
            }
        }
        const builder::Tree::Settings treeSettings = {
            .emptinessFactor = emptinessFactor,
            .traversalCost = traversalCost,
            .intersectionCost = intersectionCost,
            .maxDepth = utils::autoCast(maxDepth),
        };
        const auto progress = [start = std::chrono::steady_clock::now()](size_t progressValue)
        {
            using namespace std::chrono_literals;
            if (start + 10s < std::chrono::steady_clock::now()) {
                INVARIANT(false, "{}", progressValue);
            }
            return false;
        };
        auto tree = builder.build(treeSettings, std::make_shared<scene_data::SceneData>(std::move(sceneData)), progress);
        return tree.has_value();
    }

private:
    const builder::Builder builder{std::nullopt};
};

TEST_F(Builder, DISABLED_AllScenes)
{
    auto scenes = QDir::current().entryList(QStringList() << "*.obj", QDir::Files, QDir::Size | QDir::Reversed);
    for (const auto & sceneFileName : std::as_const(scenes)) {
        EXPECT_TRUE(buildSceneFromFile(sceneFileName));
    }
}

struct SceneFile
{
    QString sceneFileName;

    friend void PrintTo [[maybe_unused]] (const SceneFile & sceneFile, std::ostream * os)
    {
        *os << sceneFile.sceneFileName.toStdString();
    }
};

class BuilderSceneFile
    : public Builder
    , public testing::WithParamInterface<SceneFile>
{
protected:
    [[nodiscard]] bool buildSceneFromFile(const SceneFile & sceneFile)
    {
        return Builder::buildSceneFromFile(sceneFile.sceneFileName);
    }
};

// clang-format off
INSTANTIATE_TEST_SUITE_P(  // clazy:exclude=non-pod-global-static
    SimpleGeometry,
    BuilderSceneFile,
    testing::Values(
            u"pointlike_triangle.obj"_s,
            u"singularity.obj"_s,
            u"narrow_triangle.obj"_s,
            u"triangle.obj"_s,
            u"aa_triangle.obj"_s,
            u"coincident_triangles.obj"_s,
            u"aa_parallel_non_coincident_triangles.obj"_s,
            u"box.obj"_s,
            u"aa_box.obj"_s,
            u"tetrahedron.obj"_s,
            u"box_inside_box.obj"_s
        )
    );
// clang-format on

TEST_P(BuilderSceneFile, Build)
{
    EXPECT_TRUE(buildSceneFromFile(GetParam()));
}

struct SceneFileWithParams
{
    QString sceneFileName;

    float emptinessFactor = kEmptinessFactor;
    float traversalCost = kTraversalCost;
    float intersectionCost = kIntersectionCost;
    int maxDepth = kMaxdepth;

    friend void PrintTo [[maybe_unused]] (const SceneFileWithParams & sceneFileWithParams, std::ostream * os)
    {
        *os << sceneFileWithParams.sceneFileName.toStdString() << " " << sceneFileWithParams.emptinessFactor << " " << sceneFileWithParams.traversalCost << " " << sceneFileWithParams.intersectionCost << " " << sceneFileWithParams.maxDepth;
    }
};

class BuilderSceneFileWithParams
    : public Builder
    , public testing::WithParamInterface<SceneFileWithParams>
{
protected:
    [[nodiscard]] bool buildSceneFromFile(const SceneFileWithParams & param)
    {
        return Builder::buildSceneFromFile(param.sceneFileName, param.emptinessFactor, param.traversalCost, param.intersectionCost, param.maxDepth);
    }
};

// clang-format off
INSTANTIATE_TEST_SUITE_P(  // clazy:exclude=non-pod-global-static
    Fuzzed,
    BuilderSceneFileWithParams,
    testing::Values(
        SceneFileWithParams{u"test0.obj"_s, 0.285076f, 0.0657117f, 0.914504f, 123},
        SceneFileWithParams{u"test1.obj"_s, 0x1.a538900000000p-2f, 0x1.ddf3b40000000p-5f, 0x1.ecdd120000000p-4f},
        SceneFileWithParams{u"test2.obj"_s, 0.7149041295051575f, 0.060609497129917145f, 0.17161905765533447f},
        SceneFileWithParams{u"test3.obj"_s, 0.7149041295051575f, 0.18199801445007324f, 0.3812173902988434f},
        SceneFileWithParams{u"triangle_of_degenerate_triangles.obj"_s, 0x1.2222aa0000000p-1f, 0x1.5464960000000p-3f, 0x1.152a640000000p-1f}
    )
);
// clang-format on

TEST_P(BuilderSceneFileWithParams, Build)
{
    EXPECT_TRUE(buildSceneFromFile(GetParam()));
}

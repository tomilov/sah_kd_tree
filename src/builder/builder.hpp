#pragma once

#include <sah_kd_tree_fd/fwd.hpp>
#include <sah_kd_tree_fd/settings.hpp>
#include <scene_data/fwd.hpp>
#include <utils/fast_pimpl.hpp>
#include <utils/noncopyable.hpp>

#include <memory>

#include <builder/builder_export.h>

namespace builder
{

class ShareableHandle
{

};

class SAH_KD_TREE_FD_EXPORT Tree : utils::OneTime<Tree>
{
public:
    explicit Tree(const Settings & settings, const scene_data::SceneData & sceneData);
    Tree(Tree &&) noexcept;
    ~Tree();

    void build();

private:
    struct Impl;

    std::shared_ptr<Impl> impl_;

    static constexpr void completeClassContext()
    {
        checkTraits();
    }
};

}  // namespace builder

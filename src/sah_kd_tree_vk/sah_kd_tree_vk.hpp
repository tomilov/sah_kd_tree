#pragma once

#include <sah_kd_tree_vk/fwd.hpp>
#include <sah_kd_tree_vk/settings.hpp>
#include <scene_data/fwd.hpp>
#include <utils/fast_pimpl.hpp>
#include <utils/noncopyable.hpp>

#include <memory>

#include <sah_kd_tree_vk/sah_kd_tree_vk_export.h>

namespace sah_kd_tree_vk
{

class ShareableHandle
{

};

class SAH_KD_TREE_VK_EXPORT Tree : utils::OneTime<Tree>
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

}  // namespace sah_kd_tree_vk

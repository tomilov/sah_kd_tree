#pragma once

#include <builder/fwd.hpp>
#include <builder/settings.hpp>
#include <scene_data/fwd.hpp>
#include <utils/fast_pimpl.hpp>
#include <utils/noncopyable.hpp>

#include <memory>

#include <builder/builder_export.h>

namespace builder
{

class BUILDER_EXPORT ShareableHandle
{

};

class BUILDER_EXPORT Tree : utils::OneTime<Tree>
{
public:
    Tree(const Settings & settings, const scene_data::SceneData & sceneData);
    Tree(Tree &&) noexcept;
    ~Tree();

    const Settings & getSettings() const &;

    bool build();

private:
    struct Impl;

    std::shared_ptr<Impl> impl_;

    static constexpr void completeClassContext()
    {
        checkTraits();
    }
};

}  // namespace builder

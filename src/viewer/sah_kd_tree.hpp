#pragma once

#include <builder/fwd.hpp>
#include <engine/fwd.hpp>
#include <utils/noncopyable.hpp>

#include <memory>

namespace viewer
{

class Tree : utils::OneTime<Tree>
{
public:
    explicit Tree(const engine::Context & context, const builder::TreePtr & tree);
    ~Tree();

private:
    struct Impl;

    std::unique_ptr<Impl> impl_;
};

}  // namespace viewer

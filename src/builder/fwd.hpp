#pragma once

#include <memory>

namespace builder
{

struct Tree;

struct TreeDeleter
{
    void operator()(Tree * tree) const noexcept;
};

using TreePtr = std::unique_ptr<Tree, TreeDeleter>;

TreePtr makeTreePtr(Tree && tree);

}  // namespace builder

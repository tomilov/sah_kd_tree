#pragma once

#include <memory>

#include <builder/builder_export.h>

namespace builder
{

struct Tree;

struct TreeDeleter
{
    void operator()(Tree * tree) const noexcept BUILDER_EXPORT;
};

using TreePtr = std::unique_ptr<Tree, TreeDeleter>;

TreePtr makeTreePtr(Tree && tree) BUILDER_EXPORT;

}  // namespace builder

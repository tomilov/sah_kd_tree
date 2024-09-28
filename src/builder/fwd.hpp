#pragma once

#include <memory>

namespace builder
{
class Tree;

using TreePtr = std::shared_ptr<Tree>;
using TreeWeakPtr = std::weak_ptr<Tree>;
}  // namespace builder

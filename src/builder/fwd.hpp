#pragma once

#include <memory>

namespace builder
{
class Builder;
class Tree;

using TreePtr = std::shared_ptr<const Tree>;
using TreeWeakPtr = std::weak_ptr<const Tree>;
}  // namespace builder

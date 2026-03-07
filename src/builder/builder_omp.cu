#include "builder/builder.cuh"

namespace builder
{

bool TreeBuildContextOMP::build(const std::function<bool(size_t progressValue)> & progress)
{
    return true;
}

}  // namespace builder

#include <builder/builder.hpp>

#include <functional>

#include <cstddef>

namespace builder
{

class TreeBuildContextBase : public Tree
{
public:
    TreeBuildContextBase(const Settings & settings, const compute::CudaDevice & cudaDevice, const scene_data::SceneDataPtr & sceneData)
        : Tree{settings, cudaDevice, sceneData}
    {}

    virtual ~TreeBuildContextBase() = default;

    virtual bool build(const std::function<bool(size_t progressValue)> & progress) = 0;
};

class TreeBuildContextCUDA : public TreeBuildContextBase
{
public:
    using TreeBuildContextBase::TreeBuildContextBase;

    bool build(const std::function<bool(size_t progressValue)> & progress) override;
};

class TreeBuildContextTBB : public TreeBuildContextBase
{
public:
    using TreeBuildContextBase::TreeBuildContextBase;

    bool build(const std::function<bool(size_t progressValue)> & progress) override;
};

class TreeBuildContextOMP : public TreeBuildContextBase
{
public:
    using TreeBuildContextBase::TreeBuildContextBase;

    bool build(const std::function<bool(size_t progressValue)> & progress) override;
};

class TreeBuildContextCPP : public TreeBuildContextBase
{
public:
    using TreeBuildContextBase::TreeBuildContextBase;

    bool build(const std::function<bool(size_t progressValue)> & progress) override;
};

}  // namespace builder

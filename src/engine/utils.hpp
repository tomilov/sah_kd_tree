#pragma once

#include <vulkan/vulkan.hpp>

#include <array>
#include <deque>
#include <functional>
#include <iterator>
#include <vector>

#include <cstddef>

template<typename BitType>
struct std::hash<vk::Flags<BitType>>
{
    size_t operator()(vk::Flags<BitType> f) const noexcept
    {
        using MaskType = typename vk::Flags<BitType>::MaskType;
        return std::hash<MaskType>{}(static_cast<MaskType>(f));
    }
};

namespace engine
{

template<typename BitType>
class FlagBits
{
public:
    class Iterator
    {
    public:
        using iterator_category = std::input_iterator_tag;
        using value_type = BitType;

        bool operator==(const Iterator & rhs) const
        {
            return m == rhs.m;
        }

        bool operator!=(const Iterator & rhs) const
        {
            return !operator==(rhs);
        }

        BitType operator*() const
        {
            return static_cast<BitType>(m ^ (m & (m - 1)));
        }

        Iterator & operator++()
        {
            m &= m - 1;
            return *this;
        }

        Iterator operator++(int)
        {
            auto tmp = *this;
            operator++();
            return tmp;
        }

    private:
        friend FlagBits;

        using MaskType = typename vk::Flags<BitType>::MaskType;

        MaskType m = 0;

        Iterator() = default;

        explicit Iterator(vk::Flags<BitType> f)
            : m{static_cast<MaskType>(f)}
        {}
    };

    explicit FlagBits(vk::Flags<BitType> f)
        : f{f}
    {}

    static FlagBits allBits()
    {
        return FlagBits{vk::FlagTraits<BitType>::allFlags};
    }

    Iterator begin() const
    {
        return Iterator{f};
    }

    Iterator end() const
    {
        return {};
    }

private:
    const vk::Flags<BitType> f;
};

template<typename ChainHead, typename... ChainTail>
std::vector<ChainHead> toChainHeads(const std::vector<vk::StructureChain<ChainHead, ChainTail...>> & chains)
{
    std::vector<ChainHead> chainHeads;
    chainHeads.reserve(std::size(chains));
    for (const auto & chain : chains) {
        chainHeads.push_back(chain.template get<ChainHead>());
    }
    return chainHeads;
}

template<typename Head, typename... Tail, size_t N>
std::array<Head, N> getHeads(const vk::StructureChain<Head, Tail...> (&structureChains)[N])
{
    std::array<Head, N> heads;
    size_t i = 0;
    for (const vk::StructureChain<Head, Tail...> & chain : structureChains) {
        heads[i++] = chain.get();
    }
    return heads;
}

template<typename Head, typename... Tail>
std::vector<Head> getHeads(const std::vector<vk::StructureChain<Head, Tail...>> & structureChains)
{
    std::vector<Head> heads;
    heads.reserve(std::size(structureChains));
    for (const vk::StructureChain<Head, Tail...> & chain : structureChains) {
        heads.push_back(chain.get());
    }
    return heads;
}

template<vk::IndexType indexType>
using IndexCppType = typename vk::CppType<vk::IndexType, indexType>::Type;

[[nodiscard]] vk::DeviceSize alignedSize(vk::DeviceSize size, vk::DeviceSize alignment);
vk::Format indexTypeToFormat(vk::IndexType indexType);
uint32_t indexTypeRank(vk::IndexType indexType);
bool indexTypeLess(vk::IndexType lhs, vk::IndexType rhs);

}  // namespace engine

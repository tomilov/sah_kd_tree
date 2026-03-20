#pragma once

#include <utils/assert.hpp>
#include <utils/auto_cast.hpp>

#include <vulkan/vulkan.hpp>

#include <array>
#include <deque>
#include <functional>
#include <iterator>
#include <type_traits>
#include <vector>

#include <cstddef>

#include <engine/engine_export.h>

template<typename BitType>
struct std::hash<vk::Flags<BitType>>
{
    size_t operator()(vk::Flags<BitType> f) const noexcept
    {
        using MaskType = vk::Flags<BitType>::MaskType;
        return std::hash<MaskType>{}(static_cast<MaskType>(f));
    }
};

namespace engine
{

template<typename T, typename StructureChain>
struct PrependTypeToStructureChain;

template<typename T, typename... Ts>
struct PrependTypeToStructureChain<T, vk::StructureChain<Ts...>>
{
    using Type = vk::StructureChain<T, Ts...>;
};

template<typename T, typename StructureChain>
using PrependTypeToStructureChainT = PrependTypeToStructureChain<T, StructureChain>::Type;

template<typename BitType>
class FlagBits
{
    static_assert(std::is_enum_v<BitType>);
    static_assert(std::is_unsigned_v<std::underlying_type_t<BitType>>);

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

        using MaskType = vk::Flags<BitType>::MaskType;

        MaskType m = 0;

        Iterator() = default;

        explicit Iterator(vk::Flags<BitType> f)
            : m{static_cast<MaskType>(f)}
        {}
    };

    explicit FlagBits(vk::Flags<BitType> f)
        : flags{f}
    {}

    static FlagBits allBits()
    {
        return FlagBits{vk::FlagTraits<BitType>::allFlags};
    }

    Iterator begin() const
    {
        return Iterator{flags};
    }

    Iterator end() const
    {
        return {};
    }

private:
    const vk::Flags<BitType> flags;
};

template<
    typename ChainHead,
    typename... ChainTail>
std::vector<ChainHead> toChainHeads(
    const std::vector<vk::StructureChain<
        ChainHead,
        ChainTail...>> & chains)
{
    std::vector<ChainHead> chainHeads;
    chainHeads.reserve(std::size(chains));
    for (const auto & chain : chains) {
        chainHeads.push_back(chain.template get<ChainHead>());
    }
    return chainHeads;
}

template<
    typename Head,
    typename... Tail,
    size_t N>
std::array<
    Head,
    N>
getHeads(
    const vk::StructureChain<
        Head,
        Tail...> (&structureChains)[N])
{
    std::array<Head, N> heads;
    size_t i = 0;
    for (const vk::StructureChain<Head, Tail...> & chain : structureChains) {
        heads[i++] = chain.get();
    }
    return heads;
}

template<
    typename Head,
    typename... Tail>
std::vector<Head> getHeads(
    const std::vector<vk::StructureChain<
        Head,
        Tail...>> & structureChains)
{
    std::vector<Head> heads;
    heads.reserve(std::size(structureChains));
    for (const vk::StructureChain<Head, Tail...> & chain : structureChains) {
        heads.push_back(chain.get());
    }
    return heads;
}

template<
    typename Type,
    typename Head>
Type * findInPNextChain(Head * head)
{
    static_assert(vk::StructExtends<Type, Head>::value);
    ASSERT(head);
    vk::BaseOutStructure * currentStruct = utils::autoCast(const_cast<void *>(head->pNext));
    while (currentStruct) {
        if (currentStruct->sType == Type::structureType) {
            break;
        }
        currentStruct = currentStruct->pNext;
    }
    return utils::autoCast(currentStruct);
}

template<
    typename Type,
    typename Head>
const Type * findInPNextChain(const Head * head)
{
    static_assert(vk::StructExtends<Type, Head>::value);
    ASSERT(head);
    const vk::BaseInStructure * currentStruct = utils::autoCast(head->pNext);
    while (currentStruct) {
        if (currentStruct->sType == Type::structureType) {
            break;
        }
        currentStruct = currentStruct->pNext;
    }
    return utils::autoCast(currentStruct);
}

template<vk::IndexType indexType>
using IndexCppType = vk::CppType<vk::IndexType, indexType>::Type;

[[nodiscard]] vk::DeviceSize alignedSize(
    vk::DeviceSize size,
    vk::DeviceSize alignment) ENGINE_EXPORT;
[[nodiscard]] vk::Format indexTypeToFormat(vk::IndexType indexType) ENGINE_EXPORT;
[[nodiscard]] uint32_t indexTypeRank(vk::IndexType indexType) ENGINE_EXPORT;
[[nodiscard]] bool indexTypeLess(
    vk::IndexType lhs,
    vk::IndexType rhs) ENGINE_EXPORT;

}  // namespace engine

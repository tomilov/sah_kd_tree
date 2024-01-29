#pragma once

#include <utils/assert.hpp>

#include <vulkan/vulkan.hpp>

#include <bit>
#include <deque>
#include <functional>
#include <iterator>
#include <tuple>
#include <type_traits>
#include <utility>
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

        explicit Iterator(vk::Flags<BitType> f) : m{static_cast<MaskType>(f)}
        {}
    };

    explicit FlagBits(vk::Flags<BitType> f) : f{f}
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

template<typename Head, typename... Tail>
class StructureChains
{
public:
    [[nodiscard]] size_t size() const
    {
        size_t s = std::size(heads);
        ASSERT_MSG(s == std::size(tails), "!");
        return s;
    }

    void resize(size_t s)
    {
        size_t oldSize = size();
        heads.resize(s);
        tails.resize(s);
        for (size_t i = oldSize; i < s; ++i) {
            relink(i);
        }
    }

    void emplace_back()
    {
        heads.emplace_back();
        tails.emplace_back();
        relink(size() - 1);
    }

    [[nodiscard]] const std::vector<Head> & ref() const noexcept
    {
        return heads;
    }

    [[nodiscard]] std::vector<Head> & ref() noexcept
    {
        return heads;
    }

    template<typename T>
    [[nodiscard]] T & get(size_t i)
    {
        if constexpr (std::is_same_v<T, Head>) {
            return heads.at(i);
        } else {
            return std::get<T>(tails.at(i));
        }
    }

    template<typename T>
    [[nodiscard]] const T & get(size_t i) const
    {
        if constexpr (std::is_same_v<T, Head>) {
            return heads.at(i);
        } else {
            return std::get<T>(tails.at(i));
        }
    }

    template<typename T>
    [[nodiscard]] T & back()
    {
        if constexpr (std::is_same_v<T, Head>) {
            return heads.at(size() - 1);
        } else {
            return std::get<T>(tails.at(size() - 1));
        }
    }

    template<typename T>
    [[nodiscard]] const T & back() const
    {
        if constexpr (std::is_same_v<T, Head>) {
            return heads.at(size() - 1);
        } else {
            return std::get<T>(tails.at(size() - 1));
        }
    }

    template<typename U>
    void unlink(size_t i)
    {
        auto & head = heads.at(i);
        auto & tail = tails.at(i);
        auto p = &std::get<U>(tail);
        const auto next = [p]<typename H, typename... T>(const auto & next, H & h, T &... t) -> void
        {
            static_assert(sizeof...(T) != 0);
            if constexpr (std::is_same_v<H, U>) {
                INVARIANT(false, "Chain is broken or requested element is already unlinked");
            } else {
                if (h.pNext == p) {
                    h.pNext = std::exchange(p->pNext, nullptr);
                } else {
                    return next(next, t...);
                }
            }
        };
        next(next, head, std::get<Tail>(tail)...);
    }

    void unlink(size_t i)
    {
        const auto next = []<typename H, typename... T>(const auto & next, H & h, T &... t) -> void
        {
            h.pNext = nullptr;
            if (sizeof...(T) > 1) {
                return next(next, t...);
            }
        };
        return next(next, heads.at(i), std::get<Tail>(tails.at(i))...);
    }

    template<typename R>
    void relink(size_t i)
    {
        auto & head = heads.at(i);
        auto & tail = tails.at(i);
        auto p = &head.pNext;
        auto r = &std::get<R>(tail);
        if (!*p) {
            *p = r;
            return;
        }
        const auto next = [&p, r]<typename H, typename... T>(const auto & next, H & h, T &... t) -> void
        {
            if constexpr (std::is_same_v<H, R>) {
                h.pNext = std::exchange(*p, &h);
            } else {
                if (&h == *p) {
                    if (h.pNext) {
                        p = &h.pNext;
                    } else {
                        h.pNext = r;
                        return;
                    }
                } else {
                    INVARIANT(!h.pNext, "Unlinked element has non-null pNext");
                }
                return next(next, t...);
            }
        };
        return next(next, std::get<Tail>(tail)...);
    }

    void relink(size_t i)
    {
        auto & head = heads.at(i);
        auto & tail = tails.at(i);
        auto p = &head.pNext;
        const auto next = [&p]<typename H, typename... T>(const auto & next, H & h, T &... t) -> void
        {
            *p = &h;
            if constexpr (sizeof...(T) > 0) {
                p = &h.pNext;
                return next(next, t...);
            }
        };
        return next(next, head, std::get<Tail>(tail)...);
    }

private:
    std::vector<Head> heads;
    std::deque<std::tuple<Tail...>> tails;
};

template<typename Head>
class StructureChains<Head>
{
public:
    [[nodiscard]] size_t size() const
    {
        return std::size(heads);
    }

    void resize(size_t size)
    {
        heads.resize(size);
    }

    void emplace()
    {
        heads.emplace_back();
    }

    [[nodiscard]] const std::vector<Head> & ref() const noexcept
    {
        return heads;
    }

    [[nodiscard]] std::vector<Head> & ref() noexcept
    {
        return heads;
    }

    template<typename T>
    [[nodiscard]] Head & get(size_t i)
    {
        static_assert(std::is_same_v<T, Head>);
        return heads.at(i);
    }

    template<typename T>
    [[nodiscard]] const Head & get(size_t i) const
    {
        static_assert(std::is_same_v<T, Head>);
        return heads.at(i);
    }

    template<typename T>
    [[nodiscard]] Head & back()
    {
        static_assert(std::is_same_v<T, Head>);
        return heads.at(size() - 1);
    }

    template<typename T>
    [[nodiscard]] const Head & back() const
    {
        static_assert(std::is_same_v<T, Head>);
        return heads.at(size() - 1);
    }

    template<typename T>
    void unlink(size_t i) = delete;

    void unlinke() = delete;

    template<typename T>
    void relink(size_t i) = delete;

    void relink() = delete;

private:
    std::vector<Head> heads;
};

[[nodiscard]] inline vk::DeviceSize alignedSize(vk::DeviceSize size, vk::DeviceSize alignment)
{
    INVARIANT(std::has_single_bit(alignment), "Expected power of two alignment, got {:#b}", alignment);
    --alignment;
    return (size + alignment) & ~alignment;
}

}  // namespace engine

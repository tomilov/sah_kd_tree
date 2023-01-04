#pragma once

#include <utils/assert.hpp>

#include <vulkan/vulkan.hpp>

#include <deque>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <cstddef>

namespace engine
{

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
    size_t size() const
    {
        size_t s = std::size(heads);
        INVARIANT(s == std::size(tails), "!");
        return s;
    }

    void resize(size_t s)
    {
        size_t oldSize = size();
        heads.resize(s);
        tails.resize(s);
        for (size_t i = oldSize; i < s; ++i) {
            auto & head = heads[i];
            auto & tail = tails[i];
            auto p = &head.pNext;
            ((*std::exchange(p, &std::get<Tail>(tail).pNext) = &std::get<Tail>(tail)), ...);
            INVARIANT(!*p, "Expected null");
        }
    }

    const std::vector<Head> & ref() const noexcept
    {
        return heads;
    }

    std::vector<Head> & ref() noexcept
    {
        return heads;
    }

    template<typename T>
    T & get(size_t i)
    {
        if constexpr (std::is_same_v<T, Head>) {
            return heads.at(i);
        } else {
            return std::get<T>(tails.at(i));
        }
    }

    template<typename T>
    const T & get(size_t i) const
    {
        if constexpr (std::is_same_v<T, Head>) {
            return heads.at(i);
        } else {
            return std::get<T>(tails.at(i));
        }
    }

    template<typename T>
    T & back()
    {
        if constexpr (std::is_same_v<T, Head>) {
            return heads.at(size() - 1);
        } else {
            return std::get<T>(tails.at(size() - 1));
        }
    }

    template<typename T>
    const T & back() const
    {
        if constexpr (std::is_same_v<T, Head>) {
            return heads.at(size() - 1);
        } else {
            return std::get<T>(tails.at(size() - 1));
        }
    }

    template<typename T>
    void unlink(size_t i)
    {
        auto & head = heads.at(i);
        auto & tail = tails.at(i);
        auto * p = &std::get<T>(tail);
        const auto u = [p](auto & elem)
        {
            if (elem.pNext == p) {
                elem.pNext = std::exchange(p->pNext, nullptr);
            }
        };
        u(head);
        (u(std::get<Tail>(tail)), ...);
    }

    template<typename T>
    void relink(size_t i)
    {
        auto & head = heads.at(i);
        auto & tail = tails.at(i);
        auto p = &head.pNext;
        const auto r = [&p](auto & elem)
        {
            if constexpr (std::is_same_v<std::remove_reference_t<decltype(elem)>, T>) {
                elem.pNext = std::exchange(*p, &elem);
            } else if (elem.pNext) {
                p = &elem.pNext;
            }
        };
        (r(std::get<Tail>(tail)), ...);
    }

private:
    std::vector<Head> heads;
    std::deque<std::tuple<Tail...>> tails;
};

template<typename Head>
class StructureChains<Head>
{
public:
    size_t size() const
    {
        return std::size(heads);
    }

    void resize(size_t size)
    {
        heads.resize(size);
    }

    const std::vector<Head> & ref() const noexcept
    {
        return heads;
    }

    std::vector<Head> & ref() noexcept
    {
        return heads;
    }

    template<typename T>
    Head & get(size_t i)
    {
        static_assert(std::is_same_v<T, Head>);
        return heads.at(i);
    }

    template<typename T>
    const Head & get(size_t i) const
    {
        static_assert(std::is_same_v<T, Head>);
        return heads.at(i);
    }

    template<typename T>
    Head & back()
    {
        static_assert(std::is_same_v<T, Head>);
        return heads.at(size() - 1);
    }

    template<typename T>
    const Head & back() const
    {
        static_assert(std::is_same_v<T, Head>);
        return heads.at(size() - 1);
    }

    template<typename T>
    void unlink(size_t i) = delete;

    template<typename T>
    void relink(size_t i) = delete;

private:
    std::vector<Head> heads;
};

}  // namespace engine

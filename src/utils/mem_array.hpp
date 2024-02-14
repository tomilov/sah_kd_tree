#pragma once

#include <utils/fast_pimpl.hpp>

#include <memory>

#include <cstddef>

namespace utils
{

template<typename T>
class MemArray
{
public:
    MemArray() = default;

    explicit MemArray(size_t size) : size{size}, p{std::make_unique<T[]>(size)}
    {}

    [[nodiscard]] size_t getSize() const
    {
        return size;
    }

    [[nodiscard]] T * begin() &
    {
        return p.get();
    }

    [[nodiscard]] T * end() &
    {
        return p.get() + size;
    }

    [[nodiscard]] const T * begin() const &
    {
        return p.get();
    }

    [[nodiscard]] const T * end() const &
    {
        return p.get() + size;
    }

private:
    size_t size = 0;
    std::unique_ptr<T[]> p = nullptr;
};

}  // namespace utils

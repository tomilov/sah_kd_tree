#pragma once

#include <memory>
#include <stdexcept>

#include <cstddef>

namespace utils
{

template<typename T>
class MemArray
{
public:
    MemArray() = default;

    explicit MemArray(size_t sizeIn)
        : size{sizeIn}
        , p{std::make_unique<T[]>(size)}
    {}

    [[nodiscard]] bool isEmpty() const
    {
        return !p;
    }

    [[nodiscard]] size_t getCount() const
    {
        return size;
    }

    void setCount(size_t sizeIn)
    {
        if (size == sizeIn) {
            return;
        }
        this->operator=(MemArray{sizeIn});
    }

    [[nodiscard]] T * begin() &
    {
        return p.get();
    }

    [[nodiscard]] T * end() &
    {
        return p.get() + size;
    }

    [[nodiscard]] const T * cbegin() const &
    {
        return p.get();
    }

    [[nodiscard]] const T * cend() const &
    {
        return p.get() + size;
    }

    [[nodiscard]] const T * begin() const &
    {
        return cbegin();
    }

    [[nodiscard]] const T * end() const &
    {
        return cend();
    }

    [[nodiscard]] T & operator[](size_t i) &
    {
        return begin()[i];
    }

    [[nodiscard]] const T & operator[](size_t i) const &
    {
        return begin()[i];
    }

    [[nodiscard]] T & at(size_t i) &
    {
        if (i >= size) {
            throw std::out_of_range("MemArray");
        }
        return operator[](i);
    }

    [[nodiscard]] const T & at(size_t i) const &
    {
        if (i >= size) {
            throw std::out_of_range("MemArray");
        }
        return operator[](i);
    }

private:
    size_t size = 0;
    std::unique_ptr<T[]> p = nullptr;
};

template<typename T>
const T * begin(const MemArray<T> & memArray)
{
    return memArray.begin();
}

template<typename T>
const T * end(const MemArray<T> & memArray)
{
    return memArray.end();
}

template<typename T>
T * begin(MemArray<T> & memArray)
{
    return memArray.begin();
}

template<typename T>
T * end(MemArray<T> & memArray)
{
    return memArray.end();
}

}  // namespace utils

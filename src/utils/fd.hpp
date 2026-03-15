#pragma once

#include <utils/noncopyable.hpp>

#include <optional>

#include <utils/utils_export.h>

namespace utils
{

class UTILS_EXPORT Fd : utils::OneTime<Fd>
{
public:
    explicit Fd(int file);
    Fd(Fd && file) noexcept;
    ~Fd();

    [[nodiscard]] static std::optional<Fd> openDirect(const char * filepath);
    [[nodiscard]] static std::optional<Fd> createDirect(
        const char * filepath,
        int mode = 0644);
    [[nodiscard]] static std::optional<Fd> dup(int file);

    [[nodiscard]] int getFd() const &;
    [[nodiscard]] int release() &&;
    [[nodiscard]] std::optional<Fd> dup() const;

    void swap(Fd & rhs) noexcept;

private:
    int fd = -1;
};

}  // namespace utils

template struct utils::OneTime<utils::Fd>::CheckTraits;

#pragma once

#include <utils/noncopyable.hpp>

#include <utils/utils_export.h>

namespace utils
{

class UTILS_EXPORT Fd : utils::OneTime<Fd>
{
public:
    explicit Fd(int file);
    Fd(Fd && file) noexcept;
    ~Fd();

    [[nodiscard]] static Fd dup(int file);

    [[nodiscard]] const int & getFd() const &;
    [[nodiscard]] int release() &&;
    [[nodiscard]] Fd clone() const;

private:
    int fd = -1;

    static constexpr void completeClassContext [[maybe_unused]] ()
    {
        checkTraits();
    }
};

}  // namespace utils

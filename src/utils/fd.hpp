#pragma once

#include <utils/noncopyable.hpp>

#include <utils/utils_export.h>

namespace utils
{

class UTILS_EXPORT Fd : utils::OneTime<Fd>
{
public:
    Fd(Fd && file) noexcept;
    ~Fd();

    [[nodiscard]] static Fd make(int fd);
    [[nodiscard]] static Fd dup(int fd);

private:
    int fd = -1;

    explicit Fd(int fd);

    static constexpr void completeClassContext [[maybe_unused]] ()
    {
        checkTraits();
    }
};

}  // namespace utils

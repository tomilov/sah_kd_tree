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
    [[nodiscard]] static Fd dup(int file);

    [[nodiscard]] int getFd() const &;
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

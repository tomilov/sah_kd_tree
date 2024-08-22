#include <utils/assert.hpp>
#include <utils/fd.hpp>

#include <utility>

#include <unistd.h>

namespace utils
{

Fd::Fd(Fd && file) noexcept
    : fd{std::exchange(file.fd, fd)}
{
    INVARIANT(fd >= 0, "");
}

Fd::~Fd()
{
    if (fd < 0) {
        return;
    }
    ::close(fd);
}

Fd Fd::dup(int fd)
{
    return Fd{::dup(fd)};
}

int Fd::release() &&
{
    return std::exchange(fd, -1);
}

Fd::Fd(int fd)
    : fd{fd}
{}

}  // namespace utils

#include <utils/assert.hpp>
#include <utils/fd.hpp>

#include <utility>

#include <unistd.h>

namespace utils
{

Fd::Fd(int fd)
    : fd{fd}
{
    INVARIANT(fd >= 0, "{}", fd);
}

Fd::Fd(Fd && file) noexcept
    : fd{std::exchange(file.fd, fd)}
{
    INVARIANT(fd >= 0, "{}", fd);
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
    INVARIANT(fd >= 0, "{}", fd);
    fd = ::dup(fd);
    INVARIANT(fd >= 0, "{}", fd);
    return Fd{fd};
}

const int & Fd::getFd() const &
{
    return fd;
}

int Fd::release() &&
{
    return std::exchange(fd, -1);
}

Fd Fd::clone() const
{
    return Fd::dup(fd);
}

}  // namespace utils

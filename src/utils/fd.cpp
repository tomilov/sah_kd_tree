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

Fd Fd::make(int fd)
{
    return Fd{fd};
}

Fd Fd::dup(int fd)
{
    return make(::dup(fd));
}

Fd::Fd(int fd)
    : fd{fd}
{}

}  // namespace utils

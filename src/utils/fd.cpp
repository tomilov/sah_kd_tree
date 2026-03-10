#include <utils/assert.hpp>
#include <utils/fd.hpp>

#include <utility>

#include <fcntl.h>
#include <unistd.h>

namespace utils
{

Fd::Fd(int file)
    : fd{file}
{
    INVARIANT(fd >= 0, "{}", fd);
}

Fd::Fd(Fd && file) noexcept
    : fd{std::exchange(
          file.fd,
          -1)}
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

std::optional<Fd> Fd::openDirect(const char * filepath)
{
    const int file = ::open(filepath, O_RDONLY | O_DIRECT);
    if (file < 0) {
        return std::nullopt;
    }
    return Fd{file};
}

Fd Fd::dup(int file)
{
    INVARIANT(file >= 0, "{}", file);
    file = ::dup(file);
    INVARIANT(file >= 0, "{}", file);
    return Fd{file};
}

int Fd::getFd() const &
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

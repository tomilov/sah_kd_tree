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
    ASSERT_MSG(fd >= 0, "{}", fd);
}

Fd::Fd(Fd && file) noexcept
    : fd{std::exchange(
          file.fd,
          -1)}
{
    ASSERT_MSG(fd >= 0, "{}", fd);
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

std::optional<Fd> Fd::createDirect(const char * filepath)
{
    const int file = ::open(filepath, O_WRONLY | O_CREAT | O_DIRECT, 0644);
    if (file < 0) {
        return std::nullopt;
    }
    return Fd{file};
}

std::optional<Fd> Fd::dup(int file)
{
    ASSERT_MSG(file >= 0, "{}", file);
    file = ::dup(file);
    if (file < 0) {
        return std::nullopt;
    }
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

std::optional<Fd> Fd::dup() const
{
    return Fd::dup(fd);
}

void Fd::swap(Fd & rhs) noexcept
{
    std::swap(fd, rhs.fd);
}

void swap(
    Fd & lhs,
    Fd & rhs) noexcept
{
    lhs.swap(rhs);
}

}  // namespace utils

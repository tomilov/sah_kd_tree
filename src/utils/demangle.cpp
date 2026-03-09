#include <utils/demangle.hpp>

#include <memory>

#include <cstdlib>

#include <cxxabi.h>

namespace utils
{

std::string demangle(const char * mangled)
{
    int status = 0;
    std::unique_ptr<char, decltype(&std::free)> demangled{abi::__cxa_demangle(mangled, nullptr, nullptr, &status), std::free};
    return (status == 0) ? demangled.get() : mangled;
}

}  // namespace utils

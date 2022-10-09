#pragma once

#include <utils/utils_export.h>

#include <stdexcept>

namespace utils
{

class UTILS_EXPORT InvariantError : public std::runtime_error
{
public:
    using runtime_error::runtime_error;
    ~InvariantError() override = default;
};

}  // namespace utils

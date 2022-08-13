#pragma once

#include <stdexcept>

namespace utils
{

class InvariantError : public std::runtime_error
{
public:
    using runtime_error::runtime_error;
    ~InvariantError() override;
};

}  // namespace utils

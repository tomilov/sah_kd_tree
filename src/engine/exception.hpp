#pragma once

#include <stdexcept>

#include <engine/engine_export.h>

namespace engine
{

class ENGINE_EXPORT RuntimeError : public std::runtime_error
{
public:
    using runtime_error::runtime_error;
    ~RuntimeError() override = default;
};

}  // namespace engine

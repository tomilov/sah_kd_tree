#include <utils/utils.hpp>

#include <gtest/gtest.h>

using ScopeGuard = utils::ScopeGuard<void (*)()>;

static_assert(!std::is_default_constructible_v<ScopeGuard>, "one-time");
static_assert(std::is_nothrow_constructible_v<ScopeGuard, void (*)()>, "one-time");

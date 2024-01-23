#include <utils/utils.hpp>

#include <gtest/gtest.h>

using utils::ScopeGuard;

static_assert(!std::is_default_constructible_v<ScopeGuard<void (*)()>>, "one-time");
static_assert(std::is_nothrow_constructible_v<ScopeGuard<void (*)()>, void (*)()>, "one-time");
static_assert(!std::is_copy_constructible_v<ScopeGuard<void (*)()>>, "one-time");
static_assert(std::is_nothrow_move_constructible_v<ScopeGuard<void (*)()>>, "one-time");
static_assert(!std::is_copy_assignable_v<ScopeGuard<void (*)()>>, "one-time");
static_assert(!std::is_move_assignable_v<ScopeGuard<void (*)()>>, "one-time");

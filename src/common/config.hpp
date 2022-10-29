#pragma once

namespace sah_kd_tree
{
#ifdef NDEBUG
inline constexpr bool kIsDebugBuild = false;
#else
inline constexpr bool kIsDebugBuild = true;
#endif
}  // namespace sah_kd_tree

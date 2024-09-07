#pragma once

#include <functional>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

namespace engine
{

using StringUnorderedSet = std::unordered_set<const char *, std::hash<std::string_view>, std::equal_to<std::string_view>>;

template<typename T>
using StringUnorderedMultiMap = std::unordered_multimap<const char *, T, std::hash<std::string_view>, std::equal_to<std::string_view>>;

}  // namespace engine

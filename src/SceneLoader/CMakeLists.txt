find_package(assimp REQUIRED 5)
set_target_properties(
  assimp::assimp
  PROPERTIES
    MAP_IMPORTED_CONFIG_DEBUG Release
    MAP_IMPORTED_CONFIG_MINSIZEREL Release
    MAP_IMPORTED_CONFIG_RELWITHDEBINFO Release)

add_library("SceneLoader" STATIC)

target_sources(
  "SceneLoader"
  PRIVATE
    "include/SceneLoader.hpp"
    "AssimpWrappers.hpp"
    "AssimpWrappers.cpp"
    "SceneLoader.cpp")

target_include_directories(
  "SceneLoader"
  PUBLIC
    "include")

target_link_libraries(
  "SceneLoader"
  PUBLIC
    assimp::assimp
    Qt5::Core
    "Types")

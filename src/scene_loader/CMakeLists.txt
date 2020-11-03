find_package(assimp REQUIRED 5)
set_target_properties(
  assimp::assimp
  PROPERTIES
    MAP_IMPORTED_CONFIG_DEBUG Release
    MAP_IMPORTED_CONFIG_MINSIZEREL Release
    MAP_IMPORTED_CONFIG_RELWITHDEBINFO Release)

add_library("scene_loader" STATIC)

target_sources(
  "scene_loader"
  PRIVATE
    "assimpwrappers.hpp"
    "assimpwrappers.cpp"
    "include/scene_loader/scene_loader.hpp"
    "scene_loader.cpp")

target_include_directories(
  "scene_loader"
  PUBLIC
    "include"
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

target_link_libraries(
  "scene_loader"
  PUBLIC
    assimp::assimp
    Qt5::Core)

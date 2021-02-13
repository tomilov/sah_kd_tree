project("scene_loader")

find_package(assimp REQUIRED 5)
set_target_properties(
    assimp::assimp
    PROPERTIES
        MAP_IMPORTED_CONFIG_DEBUG Release
        MAP_IMPORTED_CONFIG_MINSIZEREL Release
        MAP_IMPORTED_CONFIG_RELWITHDEBINFO Release)

add_library("${PROJECT_NAME}")
generate_export_header("${PROJECT_NAME}")
target_sources(
    "${PROJECT_NAME}"
    PRIVATE
        "assimp_wrappers.hpp"
        "assimp_wrappers.cpp"
        "${PROJECT_NAME}_export.h"
        "${PROJECT_NAME}.hpp"
        "${PROJECT_NAME}.cpp")
target_link_libraries(
    "${PROJECT_NAME}"
    PRIVATE
        assimp::assimp
    PUBLIC
        Qt5::Core)

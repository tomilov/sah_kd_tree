project("scene_loader")

add_library("${PROJECT_NAME}")
generate_export_header("${PROJECT_NAME}")
target_sources(
    "${PROJECT_NAME}"
    PRIVATE
        "${PROJECT_NAME}_export.h"
        "${PROJECT_NAME}.hpp"
        "${PROJECT_NAME}.cpp"
        "assimp_wrappers.hpp"
        "assimp_wrappers.cpp")
target_link_libraries(
    "${PROJECT_NAME}"
    PRIVATE
        assimp::assimp
    PUBLIC
        Qt6::Core)

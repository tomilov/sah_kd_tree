project("scene_loader")

add_library("lib${PROJECT_NAME}")
set_target_properties(
    "lib${PROJECT_NAME}"
    PROPERTIES
        LIBRARY_OUTPUT_NAME "${PROJECT_NAME}"
        ARCHIVE_OUTPUT_NAME "${PROJECT_NAME}")
generate_export_header("lib${PROJECT_NAME}" BASE_NAME "${PROJECT_NAME}")
target_sources(
    "lib${PROJECT_NAME}"
    PRIVATE
        "${PROJECT_NAME}_export.h"
        "${PROJECT_NAME}.hpp"
        "${PROJECT_NAME}.cpp"
        "assimp_wrappers.hpp"
        "assimp_wrappers.cpp")
target_link_libraries(
    "lib${PROJECT_NAME}"
    PRIVATE
        assimp::assimp
    PUBLIC
        Qt6::Core)

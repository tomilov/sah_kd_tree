project("format")

skt_add_library(
    "${PROJECT_NAME}"
    SOURCES
        "${PROJECT_NAME}_export.h"
        "vulkan.hpp"
        "vulkan.cpp"
        "glm.hpp"
        "glm.cpp"
    PUBLIC_LINKS
        Vulkan::Headers
        "gli")

add_subdirectory("tests")

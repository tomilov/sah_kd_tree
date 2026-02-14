project("debug_utils")

skt_add_common_library(
    "${PROJECT_NAME}"
    SOURCES
        "${PROJECT_NAME}_export.h"
        "renderdoc.hpp"
        "renderdoc.cpp"
    PUBLIC_LINKS
        "libutils"
        Vulkan::Headers)

add_subdirectory("tests")

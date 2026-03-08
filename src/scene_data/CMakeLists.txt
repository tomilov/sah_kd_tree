project("scene_data")

skt_add_library(
    "${PROJECT_NAME}"
    SOURCES
        "fwd.hpp"
        "${PROJECT_NAME}_export.h"
        "${PROJECT_NAME}.hpp"
        "${PROJECT_NAME}.cpp"
    PUBLIC_LINKS
        "gli"
        "libutils"
)

skt_add_executable("${PROJECT_NAME}")

add_subdirectory("tests")

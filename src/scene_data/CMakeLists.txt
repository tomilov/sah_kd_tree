project("scene_data")

skt_add_library(
    SOURCES
        "fwd.hpp"
        "${PROJECT_NAME}_export.h"
        "${PROJECT_NAME}.hpp"
        "${PROJECT_NAME}.cpp"
    PUBLIC_LINKS
        "gli"
        "libutils"
)

skt_add_executable()

add_subdirectory("tests")

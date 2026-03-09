project("scene_loader")

include(Assimp)

skt_add_library(
    SOURCES
        "${PROJECT_NAME}_export.h"
        "${PROJECT_NAME}.hpp"
        "${PROJECT_NAME}.cpp"
        "assimp_wrappers.hpp"
        "assimp_wrappers.cpp"
    PUBLIC_LINKS
        Qt6::Core
        "libscene_data"
        "libutils"
    PRIVATE_LINKS
        assimp::assimp
)


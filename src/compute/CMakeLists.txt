project("compute")

skt_add_library(
    "${PROJECT_NAME}"
    SOURCES
        "fwd.hpp"
        "${PROJECT_NAME}_export.h"
        "${PROJECT_NAME}.hpp"
        "${PROJECT_NAME}.cpp"
        "make.hpp"
        "make.cpp"
    PUBLIC_LINKS
        CUDA::cuda_driver
        "libutils"
        "$<IF:$<BOOL:${BUILD_SHARED_LIBS}>,CUDA::cudart,CUDA::cudart_static>")

skt_add_executable("${PROJECT_NAME}")

add_subdirectory("tests")

project("compute")

skt_add_library(
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
        "$<IF:$<BOOL:${BUILD_SHARED_LIBS}>,${CUDA_cudart_LIBRARY},${CUDA_cudart_static_LIBRARY}>"
)

skt_add_executable()

add_subdirectory("tests")

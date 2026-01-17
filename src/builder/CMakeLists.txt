project("builder")

skt_add_common_library(
    "${PROJECT_NAME}"
    SOURCES
        "fwd.hpp"
        "${PROJECT_NAME}_export.h"
        "${PROJECT_NAME}.hpp"
        "${PROJECT_NAME}.cu"
    PRIVATE_LINKS
        "libsah_kd_tree"
        "libscene_loader"
        Thrust
        CUDA::cuda_driver
        "gli"
    PUBLIC_LINKS
        "libscene_data"
        "libcompute"
        "libutils"
        "$<IF:$<BOOL:${BUILD_SHARED_LIBS}>,CUDA::cudart,CUDA::cudart_static>")

skt_add_common_executable("${PROJECT_NAME}")

add_subdirectory("tests")

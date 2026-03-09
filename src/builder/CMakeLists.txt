project("builder")

skt_add_library(
    "${PROJECT_NAME}"
    SOURCES
        "fwd.hpp"
        "${PROJECT_NAME}_export.h"
        "${PROJECT_NAME}.hpp"
        "${PROJECT_NAME}.cuh"
        "${PROJECT_NAME}.cu"
        "${PROJECT_NAME}_cpp.cu"
        "${PROJECT_NAME}_omp.cu"
        "${PROJECT_NAME}_tbb.cu"
        "${PROJECT_NAME}_cuda.cu"
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
        "$<IF:$<BOOL:${BUILD_SHARED_LIBS}>,CUDA::cudart,CUDA::cudart_static>"
)

set_target_properties(
    "lib${PROJECT_NAME}"
    PROPERTIES
        UNITY_BUILD OFF
)

skt_add_executable("${PROJECT_NAME}")

add_subdirectory("tests")

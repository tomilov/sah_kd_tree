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
set(USE_DEFAULT_TRAITS "SAH_KD_TREE_BUILDER_USE_DEFAULT_TRAITS")
option(
    ${USE_DEFAULT_TRAITS}
    "Use sah_kd_tree::DefaultTraits"
    ON)
if(${USE_DEFAULT_TRAITS})
    target_compile_definitions(
        "lib${PROJECT_NAME}"
        PUBLIC
            ${USE_DEFAULT_TRAITS})
endif()

skt_add_common_executable("${PROJECT_NAME}")

add_subdirectory("tests")

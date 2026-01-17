project("soft_renderer")

skt_add_common_library(
    "${PROJECT_NAME}"
    SOURCES
        "fwd.hpp"
        "${PROJECT_NAME}_export.h"
        "${PROJECT_NAME}.hpp"
        "${PROJECT_NAME}.cpp"
        "tree.hpp"
        "tree.cu"
    PUBLIC_LINKS
        "libbuilder"
        "libutils"
        "gli"
        "$<IF:$<BOOL:${BUILD_SHARED_LIBS}>,CUDA::cudart,CUDA::cudart_static>"
    PRIVATE_LINKS
        "libsah_kd_tree"
        "libbuilder"
        "libscene_data"
        "libscene_loader"
        CUDA::cuda_driver)

skt_add_common_executable(
    "${PROJECT_NAME}"
    PRIVATE_LINKS
        "lib${PROJECT_NAME}"
        "libscene_data"
        "libscene_loader"
        "libbuilder"
        "libcompute"
        "libutils"
        "gli"
        Qt::Core)

add_subdirectory("tests")

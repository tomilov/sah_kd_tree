project("soft_renderer")

skt_add_library(
    SOURCES
        "fwd.hpp"
        "${PROJECT_NAME}_export.h"
        "${PROJECT_NAME}.hpp"
        "${PROJECT_NAME}.cpp"
        "tree.hpp"
        "tree.cu"
    PUBLIC_LINKS
        "libbuilder"
        "gli"
        "$<IF:$<BOOL:${BUILD_SHARED_LIBS}>,CUDA::cudart,CUDA::cudart_static>"
    PRIVATE_LINKS
        "libsah_kd_tree"
        "libbuilder"
        "libscene_data"
        "libscene_loader"
        CUDA::cuda_driver
        OpenMP::OpenMP_CXX
)

set_source_files_properties(
    "${PROJECT_NAME}.cpp"
    PROPERTIES
        COMPILE_OPTIONS "-O3;-ffast-math"
)

find_package(SDL3 REQUIRED)
skt_add_executable(
    LINKS
        "lib${PROJECT_NAME}"
        "libscene_data"
        "libscene_loader"
        "libbuilder"
        "libcompute"
        "gli"
        Qt::Core
        SDL3::SDL3
)

add_subdirectory("tests")

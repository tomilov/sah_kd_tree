project("compute")

skt_add_library(
    SOURCES
        "fwd.hpp"
        "${PROJECT_NAME}_export.h"
        "${PROJECT_NAME}.hpp"
        "${PROJECT_NAME}.cpp"
        "make.hpp"
        "make.cpp"
        "format.hpp"
        "format.cpp"
        "assert.hpp"
        "assert.cpp"
    PUBLIC_LINKS
        CUDA::cuda_driver
        "$<IF:$<BOOL:${BUILD_SHARED_LIBS}>,CUDA::cudart,CUDA::cudart_static>"
        "$<IF:$<BOOL:${BUILD_SHARED_LIBS}>,CUDA::cuFile,CUDA::cuFile_static>"
)

skt_add_executable()

add_subdirectory("tests")

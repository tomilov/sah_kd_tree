set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --extended-lambda")

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_EXTENSIONS ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(MSVC)
  set(CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()

find_package(OpenMP REQUIRED)

find_package(TBB REQUIRED)

add_library("${PROJECT_NAME}" STATIC)

target_compile_options(
  "${PROJECT_NAME}"
  PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/bigobj>)

target_sources(
  "${PROJECT_NAME}"
  PRIVATE
    "include/sah_kd_tree/sah_kd_tree.hpp"
    "sah_kd_tree.cuh"
    "sah_kd_tree.cu")

target_include_directories(
  "${PROJECT_NAME}"
  PUBLIC
    "include")

target_link_libraries(
  "${PROJECT_NAME}"
  PUBLIC
    CUDA::toolkit # for Thrust (or include directory CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES)
    OpenMP::OpenMP_CXX
    ${TBB_IMPORTED_TARGETS})

set_target_properties(
  "${PROJECT_NAME}"
  PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON # required for static lib
    POSITION_INDEPENDENT_CODE ON) # required if static lib will be linked into a shared lib

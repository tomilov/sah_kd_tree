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

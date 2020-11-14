add_library("${PROJECT_NAME}" STATIC)

target_compile_options(
  "${PROJECT_NAME}"
  PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/bigobj>)

target_sources(
  "${PROJECT_NAME}"
  PRIVATE
    "include/SahKdTree.hpp"
    "utility.cuh"
    "setTriangle.cu"
    "calculateTriangleBbox.cu"
    "calculateRootNodeBbox.cu"
    "generateInitialEvent.cu"
    "thinLayerNodeOffset.cu"
    "findPerfectSplit.cu"
    "selectNodeBestSplit.cu"
    "determinePolygonSide.cu"
    "getSplittedPolygonCount.cu"
    "separateSplittedPolygon.cu"
    "decoupleEventBoth.cu"
    "updatePolygonNode.cu"
    "splitPolygon.cu"
    "updateSplittedPolygonNode.cu"
    "mergeEvent.cu"
    "setNodeCount.cu"
    "splitNode.cu"
    "builder.cu")

target_include_directories(
  "${PROJECT_NAME}"
  PUBLIC
    "include")

target_link_libraries(
  "${PROJECT_NAME}"
  PUBLIC
    CUDA::toolkit # for Thrust
    OpenMP::OpenMP_CXX
    ${TBB_IMPORTED_TARGETS}
    "types")

# OMP works properly only with gcc due to nvcc uses gcc as host compiler
# and openmp flags inferred for clang cannot be passed to gcc as is
target_compile_definitions(
  "${PROJECT_NAME}"
  PUBLIC
    THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP)


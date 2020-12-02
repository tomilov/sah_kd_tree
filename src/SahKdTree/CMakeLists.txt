add_library("Types" INTERFACE)

target_sources(
  "Types"
  INTERFACE
    "include/SahKdTree/Types.hpp")

target_include_directories(
  "Types"
  INTERFACE
    "include")

add_library("SahKdTree" STATIC)

target_compile_options(
  "SahKdTree"
  PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/bigobj>)

target_sources(
  "SahKdTree"
  PRIVATE
    "include/SahKdTree.hpp"
    "Utility.cuh"
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
    "Builder.cu")

target_include_directories(
  "SahKdTree"
  PUBLIC
    "include")

target_link_libraries(
  "SahKdTree"
  PUBLIC
    CUDA::toolkit # for Thrust
    OpenMP::OpenMP_CXX
    ${TBB_IMPORTED_TARGETS}
    "Types")

# OMP works properly only with gcc due to nvcc uses gcc as host compiler
# and openmp flags inferred for clang cannot be passed to gcc as is
target_compile_definitions(
  "SahKdTree"
  PUBLIC
    "THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_${THRUST_DEVICE_SYSTEM}")

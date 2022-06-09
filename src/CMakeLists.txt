find_package(OpenMP REQUIRED)
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

find_package(TBB REQUIRED)

set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --extended-lambda")

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_EXTENSIONS ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

#set(CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS ON) # required for static lib on MSVC
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON) # required for static lib
set(CMAKE_POSITION_INDEPENDENT_CODE ON) # required if static lib will be linked into a shared lib

set(THRUST_DEVICE_SYSTEM "CUDA" CACHE STRING "Set one of CUDA, TBB, OMP, CPP as Thrust backend.")

find_package(
    Qt5
    COMPONENTS
        Core
    REQUIRED)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

add_compile_definitions(
    QT_NO_KEYWORDS
    QT_NO_FOREACH)

add_compile_definitions(
    QT_RESTRICTED_CAST_FROM_ASCII
    QT_NO_CAST_TO_ASCII
    QT_NO_CAST_FROM_BYTEARRAY
    QT_NO_NARROWING_CONVERSIONS_IN_CONNECT)

add_compile_definitions(QT_MESSAGELOGCONTEXT)

include_directories("${CMAKE_CURRENT_BINARY_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
add_compile_options(
    "-Wall"
    "-Wextra")

include(CTest)
if(BUILD_TESTING)
    find_package(GTest REQUIRED)
    include(GoogleTest)
endif()

add_subdirectory("sah_kd_tree")
add_subdirectory("scene_loader")
add_subdirectory("builder")
add_subdirectory("fuzzer")


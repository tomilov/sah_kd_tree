enable_language(CXX)

option(BUILD_SHARED_LIBS "Build shared libraries" ON)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(GenerateExportHeader)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN YES)

find_package(OpenMP REQUIRED)
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    # requires CMAKE_CUDA_HOST_COMPILER=/usr/bin/clang++
    # or just -DCMAKE_CUDA_HOST_COMPILER:FILEPATH=%{Compiler:Executable:Cxx}
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fopenmp-version=45")  # workaround for clang and OpenMP 5.0+
endif()

find_package(TBB REQUIRED)

enable_language(CUDA)

if(BUILD_SHARED_LIBS)
    set(CMAKE_CUDA_RUNTIME_LIBRARY SHARED)
else()
    set(CMAKE_CUDA_RUNTIME_LIBRARY STATIC)
endif()

set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --extended-lambda")

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS ON)

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


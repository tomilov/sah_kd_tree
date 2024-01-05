set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(GenerateExportHeader)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN YES)

if(FALSE)
    include(CheckIPOSupported)
    check_ipo_supported(
        RESULT
            ipo_is_supported
        OUTPUT
            ipo_support_check_error
        LANGUAGES
            CXX)
    if(NOT ${ipo_is_supported})
        message(WARNING "LTO is not supported: ${ipo_support_check_error}")
    endif()
endif()

include_directories(
    "${CMAKE_CURRENT_BINARY_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}")
add_compile_options(
    "-Wall"
    "-Wextra")

link_libraries(fmt::fmt)
add_compile_definitions(FMT_USE_NONTYPE_TEMPLATE_ARGS)

link_libraries(spdlog::spdlog)
add_compile_definitions($<$<CONFIG:Debug>:SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_TRACE>)

include(CUDA)
include(Thrust)
include(Qt)
include(Shaders)

include(CTest)
if(BUILD_TESTING)
    find_package(GTest REQUIRED)
    include(GoogleTest)
endif()

cmake_dependent_option(
    SAH_KD_TREE_BUILD_FUZZER
    "Whether to build 'fuzzer' target"
        ON
    "CMAKE_CXX_COMPILER_ID STREQUAL Clang"
        OFF)

add_subdirectory("common")
add_subdirectory("utils")
add_subdirectory("sah_kd_tree")
add_subdirectory("scene_loader")
add_subdirectory("builder")
add_subdirectory("fuzzer")
add_subdirectory("scene")
add_subdirectory("format")
add_subdirectory("codegen")
add_subdirectory("engine")
add_subdirectory("viewer")


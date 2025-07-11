set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(GenerateExportHeader)
# nm -gDC --defined-only lib*.so | awk 'match($2, /T/) { print substr($0, index($0, $3)) }' | less -S
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN YES)

if(TRUE)
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

option(ENABLE_IWYU "Run include-what-you-use")
if(ENABLE_IWYU)
  find_program(IWYU_EXE NAMES "include-what-you-use" REQUIRED)
  set(CMAKE_C_INCLUDE_WHAT_YOU_USE "${IWYU_EXE}")
  set(CMAKE_CXX_INCLUDE_WHAT_YOU_USE "${IWYU_EXE}")
endif()

include_directories(
    "${CMAKE_CURRENT_BINARY_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}")
add_compile_options(
    "-Wall"
    "-Wextra")
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    add_compile_options(
        -Wold-style-cast
        -Wredundant-decls
        -Wwrite-strings
        -Wcast-qual
        -Wmissing-include-dirs
        -Wsized-deallocation
        -Wduplicated-cond
        -Wcast-align
        -Wdisabled-optimization
        -Winvalid-pch
        -Woverloaded-virtual
        -Wconversion
        -Wdouble-promotion
        -Wzero-as-null-pointer-constant
        -Wuseless-cast)
endif()

link_libraries(fmt::fmt)
add_compile_definitions(FMT_USE_NONTYPE_TEMPLATE_ARGS)

link_libraries(spdlog::spdlog)
add_compile_definitions($<$<CONFIG:Debug>:SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_TRACE>)

include(Python)
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
add_subdirectory("debug_utils")
add_subdirectory("sah_kd_tree")
add_subdirectory("scene_data")
add_subdirectory("scene_loader")
add_subdirectory("compute")
add_subdirectory("builder")
add_subdirectory("fuzzer")
add_subdirectory("format")
add_subdirectory("codegen")
add_subdirectory("engine")
add_subdirectory("soft_renderer")
add_subdirectory("viewer")


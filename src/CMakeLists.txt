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
            cxx_ipo_is_supported
        OUTPUT
            cxx_ipo_support_check_error
        LANGUAGES
            CXX)
    if(NOT ${cxx_ipo_is_supported})
        message(WARNING "LTO is not supported: ${cxx_ipo_support_check_error}")
    endif()
    check_ipo_supported(
        RESULT
            cuda_ipo_is_supported
        OUTPUT
            cuda_ipo_support_check_error
        LANGUAGES
            CUDA)
    if(NOT ${cuda_ipo_is_supported})
        message(WARNING "LTO is not supported: ${cuda_ipo_support_check_error}")
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
    "-Wextra"
    $<$<COMPILE_LANGUAGE:C,CXX>:-Werror>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Werror>
    $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-copy>
    #$<$<COMPILE_LANGUAGE:CUDA>:-Wno-unused-parameter>
    #$<$<COMPILE_LANGUAGE:CUDA>:-Wno-unused-function>
    -ftemplate-backtrace-limit=0)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    add_compile_options(
        $<$<COMPILE_LANGUAGE:C,CXX>:-Wold-style-cast>
        $<$<COMPILE_LANGUAGE:C,CXX>:-Wredundant-decls>
        $<$<COMPILE_LANGUAGE:C,CXX>:-Wwrite-strings>
        $<$<COMPILE_LANGUAGE:C,CXX>:-Wcast-qual>
        $<$<COMPILE_LANGUAGE:C,CXX>:-Wmissing-include-dirs>
        $<$<COMPILE_LANGUAGE:C,CXX>:-Wsized-deallocation>
        $<$<COMPILE_LANGUAGE:C,CXX>:-Wduplicated-cond>
        $<$<COMPILE_LANGUAGE:C,CXX>:-Wcast-align>
        $<$<COMPILE_LANGUAGE:C,CXX>:-Wdisabled-optimization>
        $<$<COMPILE_LANGUAGE:C,CXX>:-Winvalid-pch>
        $<$<COMPILE_LANGUAGE:C,CXX>:-Woverloaded-virtual>
        $<$<COMPILE_LANGUAGE:C,CXX>:-Wconversion>
        $<$<COMPILE_LANGUAGE:C,CXX>:-Wdouble-promotion>
        $<$<COMPILE_LANGUAGE:C,CXX>:-Wzero-as-null-pointer-constant>
        $<$<COMPILE_LANGUAGE:C,CXX>:-Wuseless-cast>)
endif()

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

find_package(OpenMP REQUIRED)
string(REPLACE " " ";" OpenMP_CXX_FLAGS_LIST "${OpenMP_CXX_FLAGS}")
foreach(flag IN LISTS OpenMP_CXX_FLAGS_LIST)
    target_compile_options(
        OpenMP::OpenMP_CXX
        INTERFACE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${flag}>)
endforeach()

include(CUDA)
include(Thrust)

add_subdirectory("sah_kd_tree")

include(Python)
include(Qt)
include(Shaders)

link_libraries(fmt::fmt)
#add_compile_definitions(FMT_USE_NONTYPE_TEMPLATE_ARGS)

link_libraries(spdlog::spdlog)
add_compile_definitions($<$<CONFIG:Debug>:SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_TRACE>)

add_subdirectory("common")
add_subdirectory("utils")
add_subdirectory("debug_utils")
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


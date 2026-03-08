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
    -Wall
    -Wextra
    $<$<NOT:$<AND:$<CXX_COMPILER_ID:GNU>,$<COMPILE_LANGUAGE:CUDA>>>:-Wpedantic>
    $<$<COMPILE_LANGUAGE:C,CXX>:-Werror>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Werror>
    -ftemplate-backtrace-limit=0)
add_compile_options(
    $<$<COMPILE_LANGUAGE:C,CXX>:-Wold-style-cast>
    $<$<COMPILE_LANGUAGE:C,CXX>:-Wzero-as-null-pointer-constant>
    -Wcast-align
    -Wcast-qual
    -Wconversion
    -Wdisabled-optimization
    -Wdouble-promotion
    -Wformat=2
    -Winvalid-pch
    -Wmissing-include-dirs
    -Wnon-virtual-dtor
    -Woverloaded-virtual
    -Wshadow
    -Wsign-conversion
    -Wunused
    -Wwrite-strings
    -Wswitch
    -Wformat-overflow
    -Wformat-truncation
    -Wmisleading-indentation
    $<$<CXX_COMPILER_ID:GNU>:-Wduplicated-branches>
    $<$<CXX_COMPILER_ID:GNU>:-Wduplicated-cond>
    $<$<CXX_COMPILER_ID:GNU>:-Wlogical-op>
    $<$<CXX_COMPILER_ID:GNU>:-Wmaybe-uninitialized>
    $<$<CXX_COMPILER_ID:GNU>:-Wsized-deallocation>
    $<$<CXX_COMPILER_ID:GNU>:-Wuseless-cast>
    $<$<NOT:$<AND:$<CXX_COMPILER_ID:GNU>,$<NOT:$<CONFIG:Debug>>>>:-Wnull-dereference>
    $<$<NOT:$<AND:$<CXX_COMPILER_ID:GNU>,$<COMPILE_LANGUAGE:CUDA>>>:-Wredundant-decls>
    $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-copy>
    $<$<COMPILE_LANGUAGE:CUDA>:-Wno-gnu-line-marker>
    $<$<COMPILE_LANGUAGE:CUDA>:-Wno-overlength-strings>)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(
        -fstack-protector-strong
        #-fsanitize=address
        -fsanitize=undefined)
    add_link_options(
        #-fsanitize=address
        -fsanitize=undefined)

    option(ENABLE_CLANG_TIDY "Enable clang-tidy" ON)
    option(ENABLE_CPPCHECK "Enable cppcheck" ON)

    if(ENABLE_CLANG_TIDY)
        find_program(CLANG_TIDY clang-tidy REQUIRED)
        set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY}" -checks=-*,readability-*,-readability-function-cognitive-complexity,-readability-function-size,-readability-identifier-length,-readability-isolate-declaration,-readability-uppercase-literal-suffix,-readability-implicit-bool-conversion,-readability-magic-numbers,-readability-avoid-unconditional-preprocessor-if,-readability-identifier-naming,-readability-named-parameter)
    endif()

    if(ENABLE_CPPCHECK)
        find_program(CPPCHECK cppcheck REQUIRED)
        set(CMAKE_CXX_CPPCHECK "${CPPCHECK}" --enable=all --suppress=missingInclude)
    endif()
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
foreach(OpenMP_CXX_FLAG IN LISTS OpenMP_CXX_FLAGS_LIST)
    target_compile_options(
        OpenMP::OpenMP_CXX
        INTERFACE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAG}>)
endforeach()

link_libraries(fmt::fmt)

link_libraries(spdlog::spdlog)
add_compile_definitions($<$<CONFIG:Debug>:SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_TRACE>)

include(CUDA)
include(Thrust)

add_subdirectory("sah_kd_tree")

include(Python)
include(Qt)
include(Shaders)

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


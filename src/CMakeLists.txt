set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(GenerateExportHeader)
# find build/ -name 'lib*.so' | parallel -k -- nm -gDC --defined-only | awk 'match($2, /T/) { print substr($0, index($0, $3)) }' | less -S
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN YES)

include_directories(
    "${CMAKE_CURRENT_BINARY_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

add_compile_options(
    -Wall
    -Wextra
    $<$<NOT:$<AND:$<CXX_COMPILER_ID:GNU>,$<COMPILE_LANGUAGE:CUDA>>>:-Wpedantic>
    $<$<COMPILE_LANGUAGE:C,CXX>:-Werror>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Werror>
    -ftemplate-backtrace-limit=0
)
add_compile_options(
    $<$<COMPILE_LANGUAGE:C,CXX>:-Wold-style-cast>
    $<$<COMPILE_LANGUAGE:C,CXX>:-Wzero-as-null-pointer-constant>
    -Wuninitialized
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
    $<$<COMPILE_LANGUAGE:CUDA>:-Wno-overlength-strings>
)

include(Tests)

find_package(OpenMP REQUIRED)
string(REPLACE " " ";" OpenMP_CXX_FLAGS_LIST "${OpenMP_CXX_FLAGS}")
foreach(OpenMP_CXX_FLAG IN LISTS OpenMP_CXX_FLAGS_LIST)
    target_compile_options(
        OpenMP::OpenMP_CXX
        INTERFACE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAG}>
    )
endforeach()

include(CUDA)
include(Thrust)

cmake_dependent_option(
    SAH_KD_TREE_BUILD_FUZZER
    "Whether to build 'fuzzer' target"
    ON
    "CMAKE_CXX_COMPILER_ID MATCHES Clang"
    OFF
)

add_subdirectory("sah_kd_tree")

link_libraries(fmt::fmt)

link_libraries(spdlog::spdlog)
add_compile_definitions($<$<CONFIG:Debug>:SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_TRACE>)

include(Qt)
include(Python)
include(Shaders)

add_subdirectory("common")
link_libraries("libcommon")

add_subdirectory("utils")
link_libraries("libutils")

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

option(SAH_KD_TREE_GENERATE_DOCS "Whether to build the Doxygen documentation" OFF)
if(SAH_KD_TREE_GENERATE_DOCS)
    set(DOXYGEN_FILE_PATTERNS *.cpp *.cu *.inl)
    set(DOXYGEN_INCLUDE_FILE_PATTERNS *.hpp *.cuh)
    find_package(Doxygen REQUIRED COMPONENTS doxygen OPTIONAL_COMPONENTS dot dia mscgen)
    doxygen_add_docs(
        "docs"
            "${CMAKE_CURRENT_SOURCE_DIR}"
        ALL
        COMMENT "Generating API documentation with Doxygen"
    )
endif()

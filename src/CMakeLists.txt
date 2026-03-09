set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(GenerateExportHeader)
# nm -gDC --defined-only lib*.so | awk 'match($2, /T/) { print substr($0, index($0, $3)) }' | less -S
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

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_definitions(
        #_GLIBCXX_DEBUG
        #_GLIBCXX_DEBUG_PEDANTIC
        #_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_DEBUG
    )
    add_compile_options(
        -fno-omit-frame-pointer
        #-fstack-protector-strong
        #-fstack-clash-protection
        #-fcf-protection=full
        #-fsanitize=undefined
        #-fsanitize=address
        #-fsanitize=thread
        #-fsanitize=float-divide-by-zero
        #$<$<CXX_COMPILER_ID:Clang>:-fsanitize=integer>
        #$<$<CXX_COMPILER_ID:Clang>:-fsanitize=implicit-conversion>
        #$<$<CXX_COMPILER_ID:Clang>:-fsanitize=memory>
        #-fsanitize=cfi -flto -fvisibility=default
    )
    add_link_options(
        #-fsanitize=undefined
        #-fsanitize=address
        #-fsanitize=thread
        #$<$<CXX_COMPILER_ID:Clang>:-fsanitize=memory>
        #-Wl,-z,noexecstack -Wl,-z,relro -Wl,-z,now -Wl,--no-undefined
        #-fsanitize=cfi -flto
    )

    option(SAH_KD_TREE_ENABLE_CLANG_TIDY "Enable clang-tidy" OFF)
    if(SAH_KD_TREE_ENABLE_CLANG_TIDY)
        find_program(CLANG_TIDY "clang-tidy" REQUIRED)
        set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY}" --config-file=${CMAKE_SOURCE_DIR}/.clang-tidy --header-filter=${CMAKE_SOURCE_DIR}/include/.*)
    endif()

    option(SAH_KD_TREE_ENABLE_CPPCHECK "Enable cppcheck" OFF)
    if(SAH_KD_TREE_ENABLE_CPPCHECK)
        find_program(CPPCHECK "cppcheck" REQUIRED)
        set(CMAKE_CXX_CPPCHECK "${CPPCHECK}" --enable=all --suppress=missingInclude)
    endif()

    option(SAH_KD_TREE_ENABLE_IWYU "Enable include-what-you-use" OFF)
    if(SAH_KD_TREE_ENABLE_IWYU)
      find_program(IWYU_EXE NAMES "include-what-you-use" REQUIRED)
      set(CMAKE_C_INCLUDE_WHAT_YOU_USE "${IWYU_EXE}")
      set(CMAKE_CXX_INCLUDE_WHAT_YOU_USE "${IWYU_EXE}")
    endif()
else()
    add_compile_options(
        #-fstack-protector-strong
        #-D_FORTIFY_SOURCE=3
    )
    add_link_options(
        #-Wl,-z,noexecstack -Wl,-z,relro -Wl,-z,now -Wl,--no-undefined
    )
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
    "CMAKE_CXX_COMPILER_ID MATCHES Clang"
    OFF
)

find_package(OpenMP REQUIRED)
string(REPLACE " " ";" OpenMP_CXX_FLAGS_LIST "${OpenMP_CXX_FLAGS}")
foreach(OpenMP_CXX_FLAG IN LISTS OpenMP_CXX_FLAGS_LIST)
    target_compile_options(
        OpenMP::OpenMP_CXX
        INTERFACE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAG}>
    )
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


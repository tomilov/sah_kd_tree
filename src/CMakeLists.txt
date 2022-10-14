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

include_directories("${CMAKE_CURRENT_BINARY_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
add_compile_options("-Wall" "-Wextra")

if(BUILD_SHARED_LIBS)
    set(CMAKE_CUDA_RUNTIME_LIBRARY SHARED)
else()
    set(CMAKE_CUDA_RUNTIME_LIBRARY STATIC)
endif()

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --extended-lambda")

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS ON)

#set(CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS ON) # required for static lib on MSVC, on Linux results in CUDA-symbols ODR-violation
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON) # required for static lib
set(CMAKE_POSITION_INDEPENDENT_CODE ON) # required if static lib will be linked into a shared lib

# for CMAKE_CUDA_HOST_COMPILER=/usr/bin/clang++
# add -Xcompiler -fopenmp-version=45 to CMAKE_CUDA_FLAGS
# or to OpenMP_CXX_FLAGS, if Clang has OpenMP 5.0+
# and nvcc has problems with it
find_package(Thrust REQUIRED CONFIG)
thrust_create_target(Thrust FROM_OPTIONS)

find_package(assimp REQUIRED 5)
set_target_properties(
    assimp::assimp
    PROPERTIES
        MAP_IMPORTED_CONFIG_DEBUG Release
        MAP_IMPORTED_CONFIG_MINSIZEREL Release
        MAP_IMPORTED_CONFIG_RELWITHDEBINFO Release)

find_package(
    Qt6
    REQUIRED COMPONENTS
        Core
        Gui
        Widgets
        Qml
        Quick
        QuickControls2)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

add_compile_definitions(
    QT_NO_KEYWORDS
    QT_NO_FOREACH
    QT_RESTRICTED_CAST_FROM_ASCII
    QT_NO_CAST_TO_ASCII
    QT_NO_CAST_FROM_BYTEARRAY
    QT_NO_NARROWING_CONVERSIONS_IN_CONNECT
    QT_MESSAGELOGCONTEXT)

find_package(fmt REQUIRED)
link_libraries(fmt::fmt-header-only)

include(CTest)
if(BUILD_TESTING)
    find_package(GTest REQUIRED)
    include(GoogleTest)
endif()

cmake_dependent_option(
    BUILD_FUZZER
    "Whether to build 'fuzzer' target"
    ON
    "CMAKE_CXX_COMPILER_ID STREQUAL Clang"
    OFF)

find_package(Vulkan 1.3 REQUIRED)

find_program(GLSLANG_EXECUTABLE "glslangValidator" HINTS "$ENV{VULKAN_SDK}/bin")
find_program(SPIRV_OPT_EXECUTABLE "spirv-opt" HINTS "$ENV{VULKAN_SDK}/bin")
find_program(SPIRV_VAL_EXECUTABLE "spirv-val" HINTS "$ENV{VULKAN_SDK}/bin")
function(compile_shader_to_spirv shader_file)
    #cmake_path(ABSOLUTE_PATH shader_file)
    cmake_path(REPLACE_EXTENSION shader_file LAST_ONLY ".debug.spv" OUTPUT_VARIABLE debug_output_file)
    cmake_path(REPLACE_EXTENSION shader_file LAST_ONLY ".spv" OUTPUT_VARIABLE release_output_file)
    add_custom_command(
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        MAIN_DEPENDENCY "${shader_file}"
        DEPENDS "${GLSLANG_EXECUTABLE}" "${SPIRV_VAL_EXECUTABLE}"
        VERBATIM
        COMMAND "${GLSLANG_EXECUTABLE}" -q -g --target-env vulkan1.3 "${shader_file}" -o "${debug_output_file}"
        COMMAND "${SPIRV_VAL_EXECUTABLE}" --scalar-block-layout "${debug_output_file}"
        OUTPUT "${debug_output_file}")
    add_custom_command(
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        MAIN_DEPENDENCY "${debug_output_file}"
        DEPENDS "${SPIRV_OPT_EXECUTABLE}" "${SPIRV_VAL_EXECUTABLE}"
        VERBATIM
        COMMAND "${SPIRV_OPT_EXECUTABLE}" --skip-validation -O --strip-debug "${debug_output_file}" -o "${release_output_file}"
        COMMAND "${SPIRV_VAL_EXECUTABLE}" --scalar-block-layout "${release_output_file}"
        OUTPUT "${release_output_file}")
endfunction()

add_subdirectory("utils")
add_subdirectory("sah_kd_tree")
add_subdirectory("scene_loader")
add_subdirectory("builder")
add_subdirectory("fuzzer")
add_subdirectory("renderer")
add_subdirectory("viewer")


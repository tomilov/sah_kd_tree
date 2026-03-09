function(skt_env_or_default VARIABLE_NAME DEFAULT_VALUE)
    if(DEFINED ENV{${VARIABLE_NAME}})
        set(${VARIABLE_NAME} "$ENV{${VARIABLE_NAME}}" PARENT_SCOPE)
    else()
        set(${VARIABLE_NAME} "${DEFAULT_VALUE}" PARENT_SCOPE)
    endif()
endfunction()

function(skt_snake_to_camel SNAKE_STR OUTPUT_VAR)
    string(REPLACE "_" ";" PARTS "${SNAKE_STR}")
    set(RESULT "")
    foreach(PART IN LISTS PARTS)
        if(PART STREQUAL "")
            continue()
        endif()
        string(SUBSTRING "${PART}" 0 1 FIRST_CHAR)
        string(SUBSTRING "${PART}" 1 -1 REST_CHARS)
        string(TOUPPER "${FIRST_CHAR}" FIRST_CHAR_UPPER)
        string(APPEND RESULT "${FIRST_CHAR_UPPER}${REST_CHARS}")
    endforeach()
    set(${OUTPUT_VAR} "${RESULT}" PARENT_SCOPE)
endfunction()

option(SAH_KD_TREE_ENABLE_IPO "Enable IPO/LTO" OFF)
if(SAH_KD_TREE_ENABLE_IPO)
    include(CheckIPOSupported)
    check_ipo_supported(
        RESULT
            cxx_ipo_is_supported
        OUTPUT
            cxx_ipo_support_check_error
        LANGUAGES
            CUDA
    )
    if(NOT cxx_ipo_is_supported)
        message(STATUS "C++ LTO is not supported: ${cxx_ipo_support_check_error}")
    endif()
    check_ipo_supported(
        RESULT
            cuda_ipo_is_supported
        OUTPUT
            cuda_ipo_support_check_error
        LANGUAGES
            CUDA
    )
    if(NOT cuda_ipo_is_supported)
        message(STATUS "CUDA LTO is not supported: ${cuda_ipo_support_check_error}")
    endif()
endif()

function(skt_enable_target_ipo target)
    if(NOT SAH_KD_TREE_ENABLE_IPO)
        return()
    endif()
    get_target_property(sources "${target}" SOURCES)
    set(has_cxx_sources FALSE)
    set(has_cuda_sources FALSE)
    foreach(source ${sources})
        if(source MATCHES "\\.cpp$")
            set(has_cxx_sources TRUE)
        elseif(source MATCHES "\\.cu$")
            set(has_cuda_sources TRUE)
        endif()
    endforeach()
    if((NOT cxx_ipo_is_supported AND has_cxx_sources) OR (NOT cuda_ipo_is_supported AND has_cuda_sources))
        message(STATUS "LTO for ${target} is OFF")
    elseif(has_cxx_sources OR has_cuda_sources)
        set_property(TARGET "${target}" PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
        message(STATUS "LTO for ${target} is ON")
    endif()
endfunction()

function(skt_setup_target_unity_build target)
    set_target_properties(
        "${target}"
        PROPERTIES
            UNITY_BUILD_UNIQUE_ID "SAH_KD_TREE_UNITY_ID"
    )
endfunction()

function(skt_generate_export_header target base_name)
    generate_export_header("${target}" BASE_NAME "${base_name}")
    get_target_property(TARGET_TYPE "${target}" TYPE)
    if(TARGET_TYPE STREQUAL "STATIC_LIBRARY")
        string(TOUPPER "${base_name}" STATIC_DEFINE_PREFIX)
        target_compile_definitions(
            "${target}"
            PUBLIC
                ${STATIC_DEFINE_PREFIX}_STATIC_DEFINE
        )
    endif()
endfunction()

function(skt_add_library)
    cmake_parse_arguments(
        "arg"
        "INTERFACE"
        "TARGET;BASE_NAME"
        "SOURCES;PUBLIC_LINKS;PRIVATE_LINKS;SYSTEM_PUBLIC_INCLUDES"
        ${ARGN}
    )
    if(DEFINED arg_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "${PROJECT_NAME}: ${arg_UNPARSED_ARGUMENTS}")
    endif()
    if(NOT DEFINED arg_TARGET)
        set(arg_TARGET "lib${PROJECT_NAME}")
    endif()
    if(arg_INTERFACE)
        add_library("${arg_TARGET}" INTERFACE)
    else()
        add_library("${arg_TARGET}")
    endif()
    string(REGEX REPLACE "^lib" "" output_name "${arg_TARGET}")
    set_target_properties(
        "${arg_TARGET}"
        PROPERTIES
            LIBRARY_OUTPUT_NAME "${output_name}"
            ARCHIVE_OUTPUT_NAME "${output_name}"
    )
    if(NOT DEFINED arg_BASE_NAME)
        set(arg_BASE_NAME "${PROJECT_NAME}")
    endif()
    skt_generate_export_header("${arg_TARGET}" "${arg_BASE_NAME}")
    target_sources(
        "${arg_TARGET}"
        PRIVATE
            ${arg_SOURCES}
    )
    target_link_libraries(
        "${arg_TARGET}"
        PUBLIC
            ${arg_PUBLIC_LINKS}
        PRIVATE
            ${arg_PRIVATE_LINKS}
    )
    target_include_directories(
        "${arg_TARGET}"
        SYSTEM PUBLIC
            ${arg_SYSTEM_PUBLIC_INCLUDES}
    )
    skt_enable_target_ipo("${arg_TARGET}")
    skt_setup_target_unity_build("${arg_TARGET}")
endfunction()

function(skt_add_executable)
    cmake_parse_arguments(
        "arg"
        "EXTERNAL"
        "TARGET;MAIN_SOURCE;MAIN_LINK"
        "SOURCES;LINKS"
        ${ARGN}
    )
    if(DEFINED arg_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "${PROJECT_NAME}: ${arg_UNPARSED_ARGUMENTS}")
    endif()
    if(NOT DEFINED arg_TARGET)
        set(arg_TARGET "${PROJECT_NAME}")
    endif()
    if(NOT arg_EXTERNAL)
        add_executable("${arg_TARGET}")
    endif()
    if(NOT DEFINED arg_MAIN_SOURCE)
        set(arg_MAIN_SOURCE "main.cpp")
    endif()
    target_sources(
        "${arg_TARGET}"
        PRIVATE
            "${arg_MAIN_SOURCE}"
            ${arg_SOURCES}
    )
    if(NOT DEFINED arg_MAIN_LINK)
        set(arg_MAIN_LINK "lib${arg_TARGET}")
    endif()
    target_link_libraries(
        "${arg_TARGET}"
        PRIVATE
            "${arg_MAIN_LINK}"
            ${arg_LINKS}
    )
    target_compile_definitions(
        "${arg_TARGET}"
        PRIVATE
            APPLICATION_NAME="${arg_TARGET}"
    )
    skt_enable_target_ipo("${arg_TARGET}")
    skt_setup_target_unity_build("${arg_TARGET}")
endfunction()

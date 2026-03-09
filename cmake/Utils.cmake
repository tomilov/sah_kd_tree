function(skt_env_or_default variable_name default_value)
    if(DEFINED ENV{${variable_name}})
        set(${variable_name} "$ENV{${variable_name}}" PARENT_SCOPE)
    else()
        set(${variable_name} "${default_value}" PARENT_SCOPE)
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

function(skt_add_library target)
    cmake_parse_arguments(
        ARG
        "FORCE_DISABLE_IPO"
        "BASE_NAME"
        "SOURCES;PRIVATE_LINKS;PUBLIC_LINKS;SYSTEM_PUBLIC_INCLUDES"
        ${ARGN}
    )
    if(NOT ARG_BASE_NAME)
        set(ARG_BASE_NAME "${target}")
    endif()
    add_library("lib${target}")
    set_target_properties(
        "lib${target}"
        PROPERTIES
            LIBRARY_OUTPUT_NAME "${target}"
            ARCHIVE_OUTPUT_NAME "${target}"
    )
    generate_export_header("lib${target}" BASE_NAME "${ARG_BASE_NAME}")
    get_target_property(TARGET_TYPE "lib${target}" TYPE)
    if(TARGET_TYPE STREQUAL "STATIC_LIBRARY")
        string(TOUPPER "${ARG_BASE_NAME}" STATIC_DEFINE_PREFIX)
        target_compile_definitions(
            "lib${target}"
            PUBLIC
                ${STATIC_DEFINE_PREFIX}_STATIC_DEFINE
        )
    endif()
    if(ARG_SOURCES)
        target_sources(
            "lib${target}"
            PRIVATE
                ${ARG_SOURCES}
        )
    endif()
    if(ARG_PRIVATE_LINKS)
        target_link_libraries(
            "lib${target}"
            PRIVATE
                ${ARG_PRIVATE_LINKS}
        )
    endif()
    if(ARG_PUBLIC_LINKS)
        target_link_libraries(
            "lib${target}"
            PUBLIC
                ${ARG_PUBLIC_LINKS}
        )
    endif()
    if(ARG_SYSTEM_PUBLIC_INCLUDES)
        target_include_directories(
            "lib${target}"
            SYSTEM PUBLIC
                ${ARG_SYSTEM_PUBLIC_INCLUDES}
        )
    endif()
    if(NOT ARG_FORCE_DISABLE_IPO)
        skt_enable_target_ipo("lib${target}")
    endif()
    skt_setup_target_unity_build("lib${target}")
endfunction()

function(skt_add_executable target)
    cmake_parse_arguments(
        ARG
        "FORCE_DISABLE_IPO"
        ""
        "SOURCES;PRIVATE_LINKS"
        ${ARGN}
    )
    add_executable("${target}")
    target_sources(
        "${target}"
        PRIVATE
            "main.cpp"
            ${ARG_SOURCES}
    )
    target_link_libraries(
        "${target}"
        PRIVATE
            "lib${target}"
            ${ARG_PRIVATE_LINKS}
    )
    target_compile_definitions(
        "${target}"
        PRIVATE
            APPLICATION_NAME="${target}"
    )
    if(NOT ARG_FORCE_DISABLE_IPO)
        skt_enable_target_ipo("${target}")
    endif()
    skt_setup_target_unity_build("${target}")
endfunction()

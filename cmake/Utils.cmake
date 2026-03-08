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

function(skt_add_library target)
    cmake_parse_arguments(ARG "" "BASE_NAME" "SOURCES;PRIVATE_LINKS;PUBLIC_LINKS;SYSTEM_PUBLIC_INCLUDES" ${ARGN})
    if(NOT ARG_BASE_NAME)
        set(ARG_BASE_NAME "${target}")
    endif()
    add_library("lib${target}")
    set_target_properties(
        "lib${target}"
        PROPERTIES
            LIBRARY_OUTPUT_NAME "${target}"
            ARCHIVE_OUTPUT_NAME "${target}")
    generate_export_header("lib${target}" BASE_NAME "${ARG_BASE_NAME}")
    get_target_property(TARGET_TYPE "lib${target}" TYPE)
    if(TARGET_TYPE STREQUAL "STATIC_LIBRARY")
        string(TOUPPER "${ARG_BASE_NAME}" STATIC_DEFINE_PREFIX)
        target_compile_definitions(
            "lib${target}"
            PUBLIC
                ${STATIC_DEFINE_PREFIX}_STATIC_DEFINE)
    endif()
    if(ARG_SOURCES)
        target_sources(
            "lib${target}"
            PRIVATE
                ${ARG_SOURCES})
    endif()
    if(ARG_PRIVATE_LINKS)
        target_link_libraries(
            "lib${target}"
            PRIVATE
                ${ARG_PRIVATE_LINKS})
    endif()
    if(ARG_PUBLIC_LINKS)
        target_link_libraries(
            "lib${target}"
            PUBLIC
                ${ARG_PUBLIC_LINKS})
    endif()
    if(ARG_SYSTEM_PUBLIC_INCLUDES)
        target_include_directories(
            "lib${target}"
            SYSTEM PUBLIC
                ${ARG_SYSTEM_PUBLIC_INCLUDES})
    endif()
endfunction()

include(CheckIPOSupported)
check_ipo_supported(
    RESULT
        cxx_ipo_is_supported
    OUTPUT
        cxx_ipo_support_check_error
    LANGUAGES
        CXX
)
function(skt_add_executable target)
    cmake_parse_arguments(ARG "" "" "SOURCES;PRIVATE_LINKS" ${ARGN})
    add_executable("${target}")
    target_sources(
        "${target}"
        PRIVATE
            "main.cpp"
            ${ARG_SOURCES})
    target_link_libraries(
        "${target}"
        PRIVATE
            "lib${target}"
            ${ARG_PRIVATE_LINKS})
    target_compile_definitions(
        "${target}"
        PRIVATE
            APPLICATION_NAME="${target}")
    if(cxx_ipo_is_supported)
        get_target_property(sources "${target}" SOURCES)
        set(has_cuda_sources FALSE)
        foreach(source ${sources})
            if(source MATCHES "\\.cu$")
                set(has_cuda_sources TRUE)
            endif()
        endforeach()
        if(has_cuda_sources)
            set_property(TARGET "${target}" PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
            message(STATUS "LTO for ${target} is ON")
        else()
            message(STATUS "LTO for ${target} is OFF because of cuda sources")
        endif()
    endif()
endfunction()

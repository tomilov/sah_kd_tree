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

function(skt_add_library name)
    cmake_parse_arguments(ARG "" "BASE_NAME" "SOURCES;PRIVATE_LINKS;PUBLIC_LINKS;SYSTEM_PUBLIC_INCLUDES" ${ARGN})

    if(NOT ARG_BASE_NAME)
        set(ARG_BASE_NAME "${name}")
    endif()

    add_library("lib${name}")
    set_target_properties(
        "lib${name}"
        PROPERTIES
            LIBRARY_OUTPUT_NAME "${name}"
            ARCHIVE_OUTPUT_NAME "${name}")
    generate_export_header("lib${name}" BASE_NAME "${ARG_BASE_NAME}")
    get_target_property(TARGET_TYPE "lib${name}" TYPE)
    if(TARGET_TYPE STREQUAL "STATIC_LIBRARY")
        string(TOUPPER "${ARG_BASE_NAME}" STATIC_DEFINE_PREFIX)
        target_compile_definitions(
            "lib${name}"
            PUBLIC
                ${STATIC_DEFINE_PREFIX}_STATIC_DEFINE)
    endif()
    if(ARG_SOURCES)
        target_sources(
            "lib${name}"
            PRIVATE
                ${ARG_SOURCES})
    endif()
    if(ARG_PRIVATE_LINKS)
        target_link_libraries(
            "lib${name}"
            PRIVATE
                ${ARG_PRIVATE_LINKS})
    endif()
    if(ARG_PUBLIC_LINKS)
        target_link_libraries(
            "lib${name}"
            PUBLIC
                ${ARG_PUBLIC_LINKS})
    endif()
    if(ARG_SYSTEM_PUBLIC_INCLUDES)
        target_include_directories(
            "lib${name}"
            SYSTEM PUBLIC
                ${ARG_SYSTEM_PUBLIC_INCLUDES})
    endif()
endfunction()

function(skt_add_executable name)
    cmake_parse_arguments(ARG "" "" "SOURCES;PRIVATE_LINKS" ${ARGN})

    add_executable("${name}")
    target_sources(
        "${name}"
        PRIVATE
            "main.cpp"
            ${ARG_SOURCES})
    target_link_libraries(
        "${name}"
        PRIVATE
            "lib${name}"
            ${ARG_PRIVATE_LINKS})
    target_compile_definitions(
        "${name}"
        PRIVATE
            APPLICATION_NAME="${name}")
endfunction()

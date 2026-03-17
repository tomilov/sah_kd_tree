include(CTest)
if(BUILD_TESTING)
    find_package(GTest REQUIRED)
    include(GoogleTest)
endif()

function(skt_add_tests)
    if(NOT BUILD_TESTING)
        return()
    endif()
    cmake_parse_arguments(
        "arg"
        "GPU"
        "MAIN_LINK;WORKING_DIRECTORY;TIMEOUT"
        "SOURCES;LINKS"
        ${ARGN}
    )
    if(DEFINED arg_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "${PROJECT_NAME}: ${arg_UNPARSED_ARGUMENTS}")
    endif()
    if(NOT DEFINED arg_MAIN_LINK)
        set(arg_MAIN_LINK "lib${PROJECT_NAME}")
    endif()
    skt_add_executable(
        TARGET "${PROJECT_NAME}_tests"
        MAIN_SOURCE "${PROJECT_NAME}_test.cpp"
        SOURCES
            ${arg_SOURCES}
        MAIN_LINK "${arg_MAIN_LINK}"
        LINKS
            GTest::GTest
            GTest::Main
            ${arg_LINKS}
    )
    if(NOT DEFINED arg_WORKING_DIRECTORY)
        set(arg_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/data")
    endif()
    if(NOT DEFINED arg_TIMEOUT)
        set(arg_TIMEOUT 10)
    endif()
    set(extra_properties "")
    if(arg_GPU)
        list(APPEND extra_properties RESOURCE_LOCK "gpu")
    endif()
    gtest_discover_tests(
        "${PROJECT_NAME}_tests"
        WORKING_DIRECTORY
            "${arg_WORKING_DIRECTORY}"
        PROPERTIES
            TIMEOUT "${arg_TIMEOUT}"
            ${extra_properties}
    )
endfunction()

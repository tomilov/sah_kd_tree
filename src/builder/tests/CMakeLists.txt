if(NOT BUILD_TESTING)
    return()
endif()

add_executable("${PROJECT_NAME}_tests")
target_sources(
    "${PROJECT_NAME}_tests"
    PRIVATE
        "${PROJECT_NAME}_tests.cpp")
target_link_libraries(
    "${PROJECT_NAME}_tests"
    PRIVATE
        "lib${PROJECT_NAME}"
        GTest::GTest
        GTest::Main)

gtest_discover_tests(
    "${PROJECT_NAME}_tests"
    WORKING_DIRECTORY
        "${CMAKE_SOURCE_DIR}/data/scenes"
    PROPERTIES
        TIMEOUT 10)

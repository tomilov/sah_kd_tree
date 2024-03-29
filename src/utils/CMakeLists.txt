project("utils")

add_library("lib${PROJECT_NAME}")
set_target_properties(
    "lib${PROJECT_NAME}"
    PROPERTIES
        LIBRARY_OUTPUT_NAME "${PROJECT_NAME}"
        ARCHIVE_OUTPUT_NAME "${PROJECT_NAME}")
generate_export_header("lib${PROJECT_NAME}" BASE_NAME "${PROJECT_NAME}")
if(NOT BUILD_SHARED_LIBS)
    string(TOUPPER ${PROJECT_NAME} STATIC_DEFINE_PREFIX)
    target_compile_definitions(
        "lib${PROJECT_NAME}"
        PUBLIC
            ${STATIC_DEFINE_PREFIX}_STATIC_DEFINE)
endif()
target_sources(
    "lib${PROJECT_NAME}"
    PRIVATE
        "${PROJECT_NAME}_export.h"
        "${PROJECT_NAME}.hpp"
        "${PROJECT_NAME}.cpp"
        "pp.hpp"
        "meta.hpp"
        "meta.cpp"
        "noncopyable.hpp"
        "noncopyable.cpp"
        "fast_pimpl.hpp"
        "fast_pimpl.cpp"
        "overloaded.hpp"
        "overloaded.cpp"
        "assert.hpp"
        "assert.cpp"
        "exception.hpp"
        "exception.cpp"
        "auto_cast.hpp"
        "auto_cast.cpp"
        "checked_ptr.hpp"
        "checked_ptr.cpp"
        "random.hpp"
        "random.cpp"
        "scope_guard.hpp"
        "scope_guard.cpp"
        "mem_array.hpp"
        "mem_array.cpp"
        "math.hpp"
        "math.cpp")

add_subdirectory("tests")


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
        "assert.cpp"
        "assert.hpp"
        "auto_cast.cpp"
        "auto_cast.hpp"
        "checked_ptr.cpp"
        "checked_ptr.hpp"
        "exception.cpp"
        "exception.hpp"
        "fast_pimpl.cpp"
        "fast_pimpl.hpp"
        "fd.cpp"
        "fd.hpp"
        "hash.cpp"
        "hash.hpp"
        "math.cpp"
        "math.hpp"
        "mem_array.cpp"
        "mem_array.hpp"
        "meta.cpp"
        "meta.hpp"
        "noncopyable.cpp"
        "noncopyable.hpp"
        "overloaded.cpp"
        "overloaded.hpp"
        "pp.hpp"
        "random.cpp"
        "random.hpp"
        "scope_guard.cpp"
        "scope_guard.hpp")

add_subdirectory("tests")


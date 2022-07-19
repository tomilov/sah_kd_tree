project("sah_kd_tree")

add_library("lib${PROJECT_NAME}")
set_target_properties(
    "lib${PROJECT_NAME}"
    PROPERTIES
        LIBRARY_OUTPUT_NAME "${PROJECT_NAME}"
        ARCHIVE_OUTPUT_NAME "${PROJECT_NAME}")
generate_export_header("lib${PROJECT_NAME}" BASE_NAME "${PROJECT_NAME}")
target_sources(
    "lib${PROJECT_NAME}"
    PRIVATE
        "${PROJECT_NAME}_export.h"
        "${PROJECT_NAME}.cuh"
        "calculate_triangle_bbox.cu"
        "calculate_root_node_bbox.cu"
        "generate_initial_event.cu"
        "filter_layer_node_offset.cu"
        "find_perfect_split.cu"
        "select_node_best_split.cu"
        "determine_polygon_side.cu"
        "update_splitted_polygon_count.cu"
        "separate_splitted_polygon.cu"
        "decouple_event_both.cu"
        "update_polygon_node.cu"
        "split_polygon.cu"
        "update_splitted_polygon_node.cu"
        "merge_event.cu"
        "set_node_count.cu"
        "split_node.cu"
        "resize_node.cu"
        "populate_node_parent.cu"
        "populate_leaf_node_triangle_range.cu"
        "check_tree.cu"
        "calculate_rope.cu"
        "builder.cu"
        "link_triangle.cu")
if(NOT BUILD_SHARED_LIBS)
    target_compile_definitions("lib${PROJECT_NAME}" PUBLIC SAH_KD_TREE_STATIC_DEFINE)
endif()
target_link_libraries(
    "lib${PROJECT_NAME}"
    PUBLIC
        Thrust)
if(BUILD_FUZZER)
    target_compile_options(
        "lib${PROJECT_NAME}"
        PRIVATE
            "-fsanitize=fuzzer-no-link")
    target_link_options(
        "lib${PROJECT_NAME}"
        PRIVATE
            "-fsanitize=fuzzer-no-link")
    target_link_directories(
        "lib${PROJECT_NAME}"
        PUBLIC
            "/usr/lib/clang/${CMAKE_CXX_COMPILER_VERSION}/lib/linux/")
    target_link_libraries(
        "lib${PROJECT_NAME}"
        PUBLIC
            "clang_rt.fuzzer_no_main-${CMAKE_SYSTEM_PROCESSOR}")  # not needed for 'fuzzer' target, but works fine
endif()

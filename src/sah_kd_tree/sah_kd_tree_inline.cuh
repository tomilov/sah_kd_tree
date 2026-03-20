#pragma once

#include <sah_kd_tree/sah_kd_tree.cuh>

#include <sah_kd_tree/builder.cu.inl>
#include <sah_kd_tree/calculate_root_node_bbox.cu.inl>
#include <sah_kd_tree/calculate_rope.cu.inl>
#include <sah_kd_tree/calculate_triangle_bbox.cu.inl>
#include <sah_kd_tree/check_tree.cu.inl>
#include <sah_kd_tree/decouple_event_both.cu.inl>
#include <sah_kd_tree/determine_polygon_side.cu.inl>
#include <sah_kd_tree/filter_layer_node_offset.cu.inl>
#include <sah_kd_tree/find_perfect_split.cu.inl>
#include <sah_kd_tree/generate_initial_event.cu.inl>
#include <sah_kd_tree/merge_event.cu.inl>
#include <sah_kd_tree/populate_leaf_node_triangle_range.cu.inl>
#include <sah_kd_tree/populate_node_parent.cu.inl>
#include <sah_kd_tree/resize_node.cu.inl>
#include <sah_kd_tree/select_node_best_split.cu.inl>
#include <sah_kd_tree/separate_splitted_polygon.cu.inl>
#include <sah_kd_tree/set_node_count.cu.inl>
#include <sah_kd_tree/split_node.cu.inl>
#include <sah_kd_tree/split_polygon.cu.inl>
#include <sah_kd_tree/update_polygon_node.cu.inl>
#include <sah_kd_tree/update_splitted_polygon_count.cu.inl>
#include <sah_kd_tree/update_splitted_polygon_node.cu.inl>

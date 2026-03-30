#pragma once

#include <sah_kd_tree/builder.inl.cu>
#include <sah_kd_tree/calculate_root_node_bbox.inl.cu>
#include <sah_kd_tree/calculate_rope.inl.cu>
#include <sah_kd_tree/calculate_triangle_bbox.inl.cu>
#include <sah_kd_tree/check_tree.inl.cu>
#include <sah_kd_tree/decouple_event_both.inl.cu>
#include <sah_kd_tree/determine_polygon_side.inl.cu>
#include <sah_kd_tree/filter_layer_node_offset.inl.cu>
#include <sah_kd_tree/find_perfect_split.inl.cu>
#include <sah_kd_tree/generate_initial_event.inl.cu>
#include <sah_kd_tree/merge_event.inl.cu>
#include <sah_kd_tree/populate_leaf_node_triangle_range.inl.cu>
#include <sah_kd_tree/populate_node_parent.inl.cu>
#include <sah_kd_tree/resize_node.inl.cu>
#include <sah_kd_tree/select_node_best_split.inl.cu>
#include <sah_kd_tree/separate_splitted_polygon.inl.cu>
#include <sah_kd_tree/set_node_count.inl.cu>
#include <sah_kd_tree/split_node.inl.cu>
#include <sah_kd_tree/split_polygon.inl.cu>
#include <sah_kd_tree/update_polygon_node.inl.cu>
#include <sah_kd_tree/update_splitted_polygon_count.inl.cu>
#include <sah_kd_tree/update_splitted_polygon_node.inl.cu>

#include <sah_kd_tree/sah_kd_tree.cuh>

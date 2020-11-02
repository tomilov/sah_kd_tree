set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# https://quuxplusone.github.io/blog/2018/12/06/dont-use-weverything/
add_compile_options(
  "$<$<CXX_COMPILER_ID:Clang,GNU>:$<BUILD_INTERFACE:-Wall;-Wextra>>")

add_compile_definitions(
  "$<$<CXX_COMPILER_ID:MSVC>:NOMINMAX;_USE_MATH_DEFINES;_VARIADIC_MAX=10;_SCL_SECURE_NO_WARNINGS>")

add_subdirectory(sah_kd_tree)
add_subdirectory(scene_loader)
add_subdirectory(builder)

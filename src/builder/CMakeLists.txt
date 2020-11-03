add_executable("builder")

target_sources(
  "builder"
  PRIVATE
    "builder.hpp"
    "builder.cpp"
    "main.cpp")

target_link_libraries(
  "builder"
  PRIVATE
    Qt5::Core
    "${PROJECT_NAME}"
    "scene_loader")

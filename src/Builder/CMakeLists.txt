add_executable("Builder")

target_sources(
  "Builder"
  PRIVATE
    "Builder.hpp"
    "Builder.cpp"
    "main.cpp")

target_link_libraries(
  "Builder"
  PRIVATE
    Qt5::Core
    CUDA::toolkit
    "SahKdTree"
    "SceneLoader")

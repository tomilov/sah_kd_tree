add_executable("Fuzz")

target_sources(
  "Fuzz"
  PRIVATE
    "SahKdTreeFuzzer.cpp")

target_link_libraries(
  "Fuzz"
  PRIVATE
    "${PROJECT_NAME}")

target_compile_options(
  "Fuzz"
  PRIVATE
    -fsanitize=fuzzer)

target_link_libraries(
  "Fuzz"
  PRIVATE
    -fsanitize=fuzzer)

add_custom_target(
  "fuzz"
  COMMAND
    "Fuzz")

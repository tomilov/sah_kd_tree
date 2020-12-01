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
    "-fsanitize=fuzzer")

target_link_libraries(
  "Fuzz"
  PRIVATE
    "-fsanitize=fuzzer")

if(UNIX)
  add_custom_command(
    TARGET
      "Fuzz"
    POST_BUILD
    COMMAND
      ${CMAKE_COMMAND} -E create_symlink "$<TARGET_FILE:Fuzz>" "${CMAKE_SOURCE_DIR}/tools/fuzz/Fuzz"
    COMMENT
      "Created symlink to Fuzz")
endif()

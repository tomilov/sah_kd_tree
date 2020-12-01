set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(
  "$<$<CXX_COMPILER_ID:Clang,GNU>:$<BUILD_INTERFACE:-Wall;-Wextra>>")

add_compile_definitions(
  "$<$<CONFIG:Debug>:_LIBCPP_DEBUG=1;_GLIBCXX_DEBUG>"
  "$<$<CXX_COMPILER_ID:MSVC>:NOMINMAX;_USE_MATH_DEFINES;_VARIADIC_MAX=10;_SCL_SECURE_NO_WARNINGS>")

find_package(OpenMP REQUIRED)

find_package(TBB REQUIRED)

set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --extended-lambda")
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_EXTENSIONS ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(MSVC)
  set(CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON) # required for static lib
set(CMAKE_POSITION_INDEPENDENT_CODE ON) # required if static lib will be linked into a shared lib

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

add_compile_definitions(
  QT_NO_KEYWORDS
  QT_NO_FOREACH)

add_compile_definitions(
  QT_DEPRECATED_WARNINGS
  QT_FORCE_ASSERTS)

add_compile_definitions(
  QT_USE_QSTRINGBUILDER
  QT_USE_FAST_CONCATENATION
  QT_USE_FAST_OPERATOR_PLUS)

add_compile_definitions(
  QT_RESTRICTED_CAST_FROM_ASCII
  QT_NO_CAST_TO_ASCII
  QT_NO_CAST_FROM_BYTEARRAY
  QT_NO_NARROWING_CONVERSIONS_IN_CONNECT)

add_compile_definitions(QT_MESSAGELOGCONTEXT)

find_package(
  Qt5 REQUIRED
  COMPONENTS
    Core)

add_subdirectory("SahKdTree")
add_subdirectory("SceneLoader")
add_subdirectory("Builder")
if (NOT MSVC)
  option(BUILD_FUZZ "Whether to build 'Fuzz' target" ON)
  if(BUILD_FUZZ)
    add_subdirectory("Fuzz")
  endif()
endif()


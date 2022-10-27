if(BUILD_SHARED_LIBS)
    set(CMAKE_CUDA_RUNTIME_LIBRARY SHARED)
else()
    set(CMAKE_CUDA_RUNTIME_LIBRARY STATIC)
endif()

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --extended-lambda")

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS ON)

#set(CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS ON) # required for static lib on MSVC, on Linux results in CUDA-symbols ODR-violation
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON) # required for static lib
set(CMAKE_POSITION_INDEPENDENT_CODE ON) # required if static lib will be linked into a shared lib

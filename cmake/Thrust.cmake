# for CMAKE_CUDA_HOST_COMPILER=/usr/bin/clang++
# add -Xcompiler -fopenmp-version=45 to CMAKE_CUDA_FLAGS
# or to OpenMP_CXX_FLAGS, if Clang has OpenMP 5.0+
# and nvcc has problems with it
find_package(Thrust REQUIRED CONFIG)
thrust_create_target(Thrust FROM_OPTIONS)

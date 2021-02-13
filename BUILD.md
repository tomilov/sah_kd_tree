# prerequisites:
sudo pacman -S tbb openmp cuda assimp

# get sources:
git clone --recursive 'https://github.com/tomilov/sah_kd_tree'
#git clone --recursive 'https://gitee.com/tomilov/sah_kd_tree'

# configure GCC build (works with CUDA Thrust backend):
cmake -S sah_kd_tree/ -B build/ -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CUDA_ARCHITECTURES=75 -DTHRUST_DEVICE_SYSTEM=CUDA -DCMAKE_CXX_COMPILER="$( which g++ )" -DCMAKE_VERBOSE_MAKEFILE=YES
# or configure clang build (can build fuzzer):
cmake -S sah_kd_tree/ -B build/ -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CUDA_ARCHITECTURES=75 -DTHRUST_DEVICE_SYSTEM=CPP -DCMAKE_CXX_COMPILER="$( which clang++ )" -DCMAKE_VERBOSE_MAKEFILE=YES

# build:
cmake --build build/ --parallel $( nproc )

# test:
pushd build/src/
    ctest --output-on-failure --parallel $( nproc )
popd

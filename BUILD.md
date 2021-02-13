sudo pacman -S tbb openmp cuda assimp
mkdir build-sah_kd_tree
cd $_
cmake ../sah_kd_tree -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CUDA_ARCHITECTURES=75 -DTHRUST_DEVICE_SYSTEM=CPP
cmake --build . --config debug --parallel $(( $( nproc ) + 1 ))

# prerequisites:
sudo pacman -S tbb openmp cuda assimp qt6 renderdoc

# get sources:
```bash
git clone --recursive 'https://github.com/tomilov/sah_kd_tree'
#git clone --recursive 'https://gitee.com/tomilov/sah_kd_tree'
```

# configure GCC build (works with CUDA Thrust backend):
```bash
cmake -S sah_kd_tree/ -B build/ -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CUDA_ARCHITECTURES=native -DTHRUST_DEVICE_SYSTEM=CUDA -DCMAKE_CXX_COMPILER="$( which g++ )" -DCMAKE_VERBOSE_MAKEFILE=YES
```

# or configure clang build (can build fuzzer):
```bash
cmake -S sah_kd_tree/ -B build/ -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CUDA_ARCHITECTURES=native -DTHRUST_DEVICE_SYSTEM=CPP -DCMAKE_CXX_COMPILER="$( which clang++ )" -DCMAKE_VERBOSE_MAKEFILE=YES
```

# build:
```bash
cmake --build build/ --parallel $( nproc )
```

# test:
```bash
pushd build/src/
    ctest --output-on-failure --parallel $( nproc )
popd
```

# QtCreator settings for clang:
```
-DBUILD_SHARED_LIBS:BOOL=ON
-DCMAKE_CUDA_ARCHITECTURES:STRING=native
-DCMAKE_CUDA_FLAGS:UNINITIALIZED=
-DCMAKE_CUDA_HOST_COMPILER:FILEPATH=%{Compiler:Executable:Cxx}
-DCMAKE_CXX_COMPILER:FILEPATH=%{Compiler:Executable:Cxx}
-DCMAKE_CXX_FLAGS:STRING=-march=native
-DCMAKE_C_COMPILER:FILEPATH=%{Compiler:Executable:C}
-DCMAKE_C_FLAGS:STRING=-march=native
-DCMAKE_PREFIX_PATH:PATH=%{Qt:QT_INSTALL_PREFIX}
-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON
-DQT_QMAKE_EXECUTABLE:FILEPATH=%{Qt:qmakeExecutable}
-DTHRUST_DEVICE_SYSTEM:STRING=CUDA
```

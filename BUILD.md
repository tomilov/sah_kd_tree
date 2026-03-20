# prerequisites:
pacman -S \
    base-devel \
    cmake \
    git \
    python3 \
    cuda \
    glslang \
    vulkan-headers \
    gtest \
    renderdoc \
    assimp \
    qt6 \
    tbb \
    openmp \
    doxygen \
    sdl3 \
    graphviz

# get sources:
```bash
git clone --recursive https://github.com/tomilov/sah_kd_tree
```

# configure:
```bash
cmake -S sah_kd_tree/ -B build/
```

# build:
```bash
cmake --build build/ --parallel
```

# test:
```bash
ctest --test-dir build/src/ --output-on-failure --parallel
```
or even:
```bash
make docker-run COMMAND=test-cpp
```

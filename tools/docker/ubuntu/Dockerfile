# syntax=docker/dockerfile:1

FROM nvidia/cuda:11.7.0-devel-ubuntu22.04

RUN apt-get update
RUN apt-get install -y wget
RUN wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key --keyring /etc/apt/trusted.gpg.d/llvm.gpg add -
RUN echo 'deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy main' >/etc/apt/sources.list.d/llvm.list
RUN apt-get update
RUN apt-get install -y \
        nvidia-cuda-dev \
        git cmake cmake-extras \
        clang-13 lld-13 libc++-13-dev libc++abi-13-dev libomp-13-dev \
        libz-dev \
        libfmt-dev \
        libtbb-dev \
        qt6-base-dev \
        libassimp-dev \
        libgtest-dev
RUN echo 'set_target_properties(assimp::assimp PROPERTIES IMPORTED_LOCATION_RELEASE "/usr/lib/x86_64-linux-gnu/libassimp.so")' \
        >>/usr/lib/x86_64-linux-gnu/cmake/assimp-5.2/assimpConfig.cmake

WORKDIR ~/sah_kd_tree/
RUN git init
RUN git remote add origin https://github.com/tomilov/sah_kd_tree
RUN git fetch
RUN git checkout -b develop origin/develop
RUN bash -ic 'make BUILD_DIR=../build-sah_kd_tree/ BUILD_SHARED_LIBS=ON LINKER=/usr/bin/lld-13 CC=/usr/bin/clang-13 CXX=/usr/bin/clang++-13 THRUST_DEVICE_SYSTEM=CUDA build'
RUN bash -ic 'history -s ../build-sah_kd_tree/bin/builder data/scenes/triangle.obj'

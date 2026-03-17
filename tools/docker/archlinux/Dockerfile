# syntax = docker/dockerfile:labs

FROM archlinux:latest

SHELL ["/bin/bash", "-xue", "-o", "pipefail", "-O", "lastpipe", "-c"]

RUN \
  --mount=type=cache,sharing=locked,target=/var/cache/pacman \
  <<EOF
pacman -Syu --noconfirm --needed \
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
    graphviz \
    clang \
    libc++
EOF

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV CUDA_PATH=/opt/cuda
ENV CMAKE_PREFIX_PATH=/opt/cuda/targets/x86_64-linux/lib/cmake

#!/usr/bin/env bash

set -xue -o pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
docker build \
    --progress plain \
    --tag $USER-sah_kd_tree:latest \
    --load \
    $SCRIPT_DIR

SRC_DIR=$( realpath $SCRIPT_DIR/../../.. )
VENV_DIR=/tmp/$USER-sah_kd_tree/venv
BUILD_DIR=/tmp/$USER-sah_kd_tree/build
mkdir -p $VENV_DIR $BUILD_DIR
docker run \
    -it \
    --name $USER-sah_kd_tree \
    --rm \
    --gpus all \
    --user $( id -u ):$( id -g ) \
    --env CXX \
    --env CXXFLAGS \
    --env CUDACXX \
    --env CUDAFLAGS \
    --env CUDAARCHS \
    --env CUDAHOSTCXX \
    --mount type=bind,src=$SRC_DIR,dst=/sah_kd_tree \
    --mount type=bind,src=$VENV_DIR,dst=/sah_kd_tree/venv \
    --mount type=bind,src=$BUILD_DIR,dst=/sah_kd_tree/build \
    --workdir /sah_kd_tree \
    $USER-sah_kd_tree:latest \
    make $@

#!/usr/bin/env bash

set -xue -o pipefail

NAME=$USER-sah_kd_tree

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
docker build \
    --progress plain \
    --tag $NAME:latest \
    --load \
    $SCRIPT_DIR

SRC_DIR=$( realpath $SCRIPT_DIR/../../.. )
VENV_DIR=/tmp/$NAME/venv
BUILD_DIR=/tmp/$NAME/build
CACHE_DIR=/tmp/$NAME/cache
CMAKE_DIR=/tmp/$NAME/cmake
mkdir -p $VENV_DIR $BUILD_DIR $CACHE_DIR $CMAKE_DIR
docker run \
    -it \
    --name $NAME \
    --hostname $NAME \
    --rm \
    --gpus all \
    --user $( id -u ):$( id -g ) \
    --mount type=bind,src=$SRC_DIR,dst=/sah_kd_tree \
    --mount type=bind,src=$VENV_DIR,dst=/sah_kd_tree/venv \
    --mount type=bind,src=$BUILD_DIR,dst=/sah_kd_tree/build \
    --mount type=bind,src=$CACHE_DIR,dst=/.cache \
    --mount type=bind,src=$CMAKE_DIR,dst=/.cmake \
    --workdir /sah_kd_tree \
    $NAME:latest \
    make $@

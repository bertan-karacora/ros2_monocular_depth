#!/usr/bin/env bash

set -euo pipefail

readonly path_repo="$(dirname "$(dirname "$(realpath "$BASH_SOURCE")")")"
source "$path_repo/libs/ros2_config/env.sh"

readonly name_container="$NAME_CONTAINER_ROS2_MONOCULAR_DEPTH"
use_clean=""
use_debug=""

show_help() {
    echo "Usage:"
    echo "  ./build.sh [-h | --help] [--use_clean] [--use_debug]"
    echo
    echo "Build the container image."
    echo
}

parse_args() {
    local arg=""
    while [[ "$#" -gt 0 ]]; do
        arg="$1"
        shift
        case $arg in
        -h | --help)
            show_help
            exit 0
            ;;
        --use_clean)
            use_clean=1
            ;;
        --use_debug)
            use_debug=1
            ;;
        *)
            echo "Unknown option $arg"
            exit 1
            ;;
        esac
    done
}

build() {
    local arch="$(arch)"

    docker build \
        --build-arg=USER \
        --build-arg UID="$UID" \
        --build-arg=DISTRIBUTION_ROS \
        --tag="$name_container:$arch" \
        --file="$path_repo/container/Dockerfile" \
        ${use_clean:+--no-cache} \
        ${use_debug:+--progress=plain} \
        "$path_repo"
}

main() {
    parse_args "$@"
    build
}

main "$@"
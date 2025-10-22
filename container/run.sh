#!/usr/bin/env bash

set -euo pipefail

readonly path_repo="$(dirname "$(dirname "$(realpath "$BASH_SOURCE")")")"
source "$path_repo/libs/ros2_config/env.sh"

readonly name_container="$NAME_CONTAINER_ROS2_MONOCULAR_DEPTH"
command=""
use_detach=1
is_found_overlay_ros2_config=""
is_found_overlay_ros2_utils=""

show_help() {
    echo "Usage:"
    echo "  ./run.sh [-h | --help] [-a | --use_attach] [<command>]"
    echo
    echo "Run the container."
    echo
}

parse_args() {
    local arg=""
    while [[ "$#" -gt 0 ]]; do
        arg="$1"
        shift
        case "$arg" in
        -h | --help)
            show_help
            exit 0
            ;;
        -a | --use_attach)
            use_detach=""
            ;;
        *)
            if [[ -z "$command" ]]; then
                command="$arg"
            else
                command="$command $arg"
            fi
            ;;
        esac
    done
}

check_overlays() {
    if [[ ! -z "$PATH_ROS2_CONFIG" ]]; then
        if [[ -d "$PATH_ROS2_CONFIG" ]]; then
            is_found_overlay_ros2_config=1
            echo "Overlaying repo at $PATH_ROS2_CONFIG"
        else
            echo "Repo at $PATH_ROS2_CONFIG not found"
        fi
    fi
    if [[ ! -z "$PATH_ROS2_UTILS" ]]; then
        if [[ -d "$PATH_ROS2_UTILS" ]]; then
            is_found_overlay_ros2_utils=1
            echo "Overlaying repo at $PATH_ROS2_UTILS"
        else
            echo "Repo at $PATH_ROS2_UTILS not found"
        fi
    fi
}

run_container() {
    local arch="$(arch)"
    local name_repo="$(basename "$path_repo")"

    docker run \
        --name "$name_container" \
        --shm-size 12G \
        --gpus all \
        --ipc host \
        --interactive \
        --tty \
        --net host \
        --rm \
        --env DISPLAY \
        ${use_detach:+"--detach"} \
        --volume /etc/localtime:/etc/localtime:ro \
        --volume /tmp/.X11-unix/:/tmp/.X11-unix/:ro \
        --volume "$HOME/.Xauthority:/home/$USER/.Xauthority:ro" \
        --volume "$HOME/.ros/:/home/$USER/.ros/" \
        --volume "$path_repo:/home/$USER/colcon_ws/src/$name_repo" \
        --volume "$HOME/data/ScanNet:/home/$USER/colcon_ws/src/$name_repo/data/ScanNet" \
        --volume "$HOME/data/ARKitScenes:/home/$USER/colcon_ws/src/$name_repo/data/ARKitScenes" \
        ${is_found_overlay_ros2_config:+--volume "$PATH_ROS2_CONFIG:/home/$USER/colcon_ws/src/$name_repo/libs/ros2_config"} \
        ${is_found_overlay_ros2_utils:+--volume "$PATH_ROS2_UTILS:/home/$USER/colcon_ws/src/ros2_utils"} \
        "$name_container:$arch" \
        ${command:+"$command"}
}

main() {
    parse_args "$@"
    check_overlays
    run_container
}

main "$@"

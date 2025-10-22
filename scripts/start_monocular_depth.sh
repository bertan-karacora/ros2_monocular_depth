#!/usr/bin/env bash

set -eo pipefail

readonly path_repo="$(dirname "$(dirname "$(realpath "$BASH_SOURCE")")")"
source "$path_repo/libs/ros2_config/env.sh"

source "/opt/ros/$DISTRIBUTION_ROS/setup.bash"
source "$HOME/colcon_ws/install/setup.bash"

set -u

args_monocular_depth=""

show_help() {
    echo "Usage:"
    echo "  ./start_monocular_depth.sh [-h | --help] [<args_monocular_depth>]"
    echo
    echo "Start monocular_depth."
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
        *)
            if [[ -z "$args_monocular_depth" ]]; then
                args_monocular_depth="$arg"
            else
                args_monocular_depth="$args_monocular_depth $arg"
            fi
            ;;
        esac
    done
}

start_monocular_depth() {
    # TODO: Use launchfile and args
    ros2 run ros2_monocular_depth spin_monocular_depth
}

main() {
    parse_args "$@"
    start_monocular_depth
}

main "$@"

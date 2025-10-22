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
    echo "  ./start_all.sh [-h | --help] [<args_monocular_depth>]"
    echo
    echo "Start all processes."
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

start_tmux_monocular_depth() {
    local path_log="$HOME/.ros/log/$NAME_CONTAINER_ROS2_MONOCULAR_DEPTH.log"

    if [ -f "$path_log" ]; then
        >"$path_log"
        echo "Reset log file $path_log"
    else
        touch "$path_log"
        echo "Created log file $path_log"
    fi

    tmux new -d -s monocular_depth "$path_repo/scripts/start_monocular_depth.sh" ${args_monocular_depth:+$args_monocular_depth}
    tmux pipe-pane -t monocular_depth -o "cat >> $path_log"
}

attach_to_tmux_monocular_depth() {
    tmux a -t monocular_depth
}

main() {
    parse_args "$@"

    "$path_repo/scripts/download_weights_pretrained.sh" "$path_repo/resources/weights"
    "$path_repo/scripts/download_data.sh" "$path_repo/resources/data"

    # TODO: Build for all models
    # Evaluation in paper is performed on (3, 518, <some_multiple_of_14>) or (3, <some_multiple_of_14>, 518) with 518 for the smaller dimension and while keeping the original aspect ratio
    # Paper states that inference works on other image scales as well (basically as long as dims are multiples of 14), obviously with different performance
    python "$path_repo/ros2_monocular_depth/scripts/export_model.py" \
        --weights "depth_anything_v2_metric_hypersim_vits" \
        --height 406 \
        --width 532 \
        --min_size_resized 400 \
        --base 14
    python "$path_repo/ros2_monocular_depth/scripts/build_engine.py" \
        --model "depth_anything_v2_metric_hypersim_vits" \
        --limit_memory_workspace "$((1 << 32))"

    start_tmux_monocular_depth
    attach_to_tmux_monocular_depth
}

main "$@"

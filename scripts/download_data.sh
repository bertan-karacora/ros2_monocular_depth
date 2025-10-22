#!/usr/bin/env bash

set -euo pipefail

readonly path_repo="$(dirname "$(dirname "$(realpath "$BASH_SOURCE")")")"
source "$path_repo/libs/ros2_config/config.sh"

path_dir_data=""

show_help() {
    echo "Usage:"
    echo "  ./download_data.sh <path_dir_data>"
    echo
    echo "Download data to <path_dir_data>."
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
        *)
            if [[ -z "$path_dir_data" ]]; then
                path_dir_data="$arg"
            else
                echo "Unknown option $arg"
                exit 1
            fi
            ;;
        esac
    done
}

download_data() {
    if [ ! -d "$path_dir_data" ]; then
        mkdir -p "$path_dir_data"
    fi

    # TODO
}

main() {
    parse_args "$@"
    download_data
}

main "$@"

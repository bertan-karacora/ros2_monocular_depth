#!/usr/bin/env bash

main() {
    # Install this at run-time as compilers are not available on all platforms at build-time
    pip install --no-cache-dir /opt/torch2trt

    exec "$@"
}

main "$@"

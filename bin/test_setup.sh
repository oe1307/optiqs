#!/bin/sh

failed_versions=""

# Check every supported Python version, including the minimum version.
for v in 3.9 3.10 3.11 3.12 3.13 3.14 3.15; do
    echo "=== Python $v ==="

    if docker build --build-arg PYTHON_VERSION="$v" -t "optiqs-test:$v" . && docker run --rm "optiqs-test:$v"; then
        echo "OK: Python $v"
    else
        echo "NG: Python $v"
        failed_versions="${failed_versions:+$failed_versions }$v"
    fi
done

if [ -n "$failed_versions" ]; then
    echo "Failed Python versions: $failed_versions" >&2
    exit 1
fi

echo "All Python versions passed."
exit 0

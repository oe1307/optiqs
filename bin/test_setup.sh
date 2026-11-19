for v in 3.11 3.12 3.13 3.14 3.15; do
    echo "=== Python $v ==="

    if docker build --build-arg PYTHON_VERSION="$v" -t "optiqs-test:$v" . && docker run --rm "optiqs-test:$v"; then
        echo "OK: Python $v"
    else
        echo "NG: Python $v"
    fi
done

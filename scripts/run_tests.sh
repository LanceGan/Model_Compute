#!/bin/bash
set -e
echo "=== Running C++ tests ==="
cd build && ctest --output-on-failure
cd ..
echo "=== Running Python tests ==="
python -m pytest python/tests/ -v
echo "=== All tests passed ==="

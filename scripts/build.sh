#!/bin/bash
set -e
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release -j$(nproc 2>/dev/null || echo 4)
echo "Build complete. Python module at: build/cpp/"

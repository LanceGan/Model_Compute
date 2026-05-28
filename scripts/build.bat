@echo off
REM Build script for Windows (PowerShell or CMD)
REM Uses the current conda/virtualenv Python for pybind11 module compilation.
setlocal

echo === Using Python: %PYTHON% ===
if "%PYTHON%"=="" (
    set PYTHON=python
)

for /f "delims=" %%i in ('%PYTHON% -c "import sys; print(sys.executable)"') do set PYTHON_EXEC=%%i
echo === Python executable: %PYTHON_EXEC% ===

echo === Creating build directory ===
if not exist build mkdir build

echo === Running CMake configure ===
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE=%PYTHON_EXEC% -DPYTHON_EXECUTABLE=%PYTHON_EXEC%
if errorlevel 1 (
    echo CMake configure failed!
    exit /b 1
)

echo === Building ===
cmake --build build --config Release
if errorlevel 1 (
    echo Build failed!
    exit /b 1
)

echo === Build complete. Python module at: build\cpp\ ===
endlocal

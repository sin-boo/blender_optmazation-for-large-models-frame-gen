#!/bin/bash

echo "Starting Screen Capture Application..."
echo

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Find Python
PYTHON_CMD=""
if command_exists python3; then
    PYTHON_CMD="python3"
elif command_exists python; then
    # Check if it's Python 3
    if python -c 'import sys; exit(0 if sys.version_info[0] == 3 else 1)' 2>/dev/null; then
        PYTHON_CMD="python"
    fi
fi

if [ -z "$PYTHON_CMD" ]; then
    echo "ERROR: Python 3 not found!"
    echo "Please install Python 3.7+ using your package manager:"
    echo
    echo "  macOS: brew install python3"
    echo "  Ubuntu/Debian: sudo apt install python3 python3-pip"
    echo "  CentOS/RHEL: sudo yum install python3 python3-pip"
    echo "  Arch: sudo pacman -S python python-pip"
    echo
    exit 1
fi

echo "Using Python: $PYTHON_CMD ($(${PYTHON_CMD} --version))"
echo

# Check Python version
if ! ${PYTHON_CMD} -c 'import sys; exit(0 if sys.version_info >= (3, 7) else 1)' 2>/dev/null; then
    echo "ERROR: Python 3.7 or higher is required!"
    echo "Current version: $(${PYTHON_CMD} --version)"
    exit 1
fi

# Make sure we're in the script directory
cd "$(dirname "$0")"

# Run the launcher
${PYTHON_CMD} launcher.py

# Check exit code
if [ $? -ne 0 ]; then
    echo
    echo "Application exited with error code $?"
    if command_exists read; then
        echo "Press Enter to continue..."
        read
    fi
fi

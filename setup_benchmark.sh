#!/bin/bash
# Script to set up the Python environment for benchmarking

# Make script executable
chmod +x benchmark.py

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3 and try again."
    exit 1
fi

# Check for pip
if ! command -v pip3 &> /dev/null; then
    echo "pip3 is not installed. Attempting to install..."
    sudo apt-get update
    sudo apt-get install -y python3-pip
fi

# Install required Python packages
echo "Installing required Python packages..."
pip3 install numpy matplotlib pandas

# Make sure the benchmark script is executable
chmod +x benchmark.py

echo "Setup complete. You can now run the benchmark with:"
echo "./benchmark.py" 
#!/bin/bash

# =======================================================================================
# Script Name:    setup_ollama_gpu.sh
# Description:    Automates the installation of Docker and the NVIDIA Container Toolkit 
#                 (pinned to v1.18.1), then launches Ollama with GPU support.
# Author:         Egor Alekseyevich Morozov
# Date:           2025-01-28
# Version:        1.1
# Requirements:   - Ubuntu/Debian-based Linux distribution
#                 - Sudo/Root privileges
#                 - NVIDIA GPU with proprietary drivers installed
# =======================================================================================

# Exit immediately if a command exits with a non-zero status
set -e

# ==========================================
# 1. Check and Install Docker
# ==========================================
echo "--- Step 1: Checking Docker Installation ---"
if ! command -v docker &> /dev/null; then
    echo "Docker not found. Installing via get.docker.com..."
    curl -fsSL https://get.docker.com | sh
    
    # Start Docker and enable it to start on boot
    sudo systemctl enable --now docker
else
    echo "Docker is already installed. Skipping installation."
fi

# ==========================================
# 2. Install NVIDIA Container Toolkit (Pinned Version)
# ==========================================
echo "--- Step 2: Installing NVIDIA Container Toolkit ---"

# Update package list and install dependencies
sudo apt-get update && sudo apt-get install -y --no-install-recommends \
   curl \
   gnupg2

# Setup NVIDIA GPG Key and Repository
# Added --yes to gpg to allow script re-runs without prompt
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Enable experimental packages support
sudo sed -i -e '/experimental/ s/^#//g' /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update

# Set specific version variable
export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.18.1-1

# Install specific version packages
sudo apt-get install -y \
    nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
    nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
    libnvidia-container-tools=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
    libnvidia-container1=${NVIDIA_CONTAINER_TOOLKIT_VERSION}

# ==========================================
# 3. Configure Docker Runtime and Restart
# ==========================================
echo "--- Step 3: Configuring Docker Runtime ---"
sudo nvidia-ctk runtime configure --runtime=docker

echo "Restarting Docker..."
sudo systemctl restart docker

# Wait a moment for Docker daemon to fully initialize
echo "Waiting for Docker daemon to initialize..."
sleep 5

echo "Installation and configuration complete!"

# ==========================================
# 4. Verification Step
# ==========================================
echo "--- Step 4: Verification ---"
echo "Verifying NVIDIA Container Toolkit installation with a test container..."
# Using a slightly newer CUDA base tag to ensure compatibility with modern cards
sudo docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi

# ==========================================
# 5. Run Ollama inside a Docker container
# ==========================================
echo "--- Step 5: Launching Ollama ---"
echo "Running Ollama inside a Docker container..."

# Added 'sudo' here for consistency. 
# If the user is not in the 'docker' group, running without sudo would fail.
sudo docker run -d --gpus=all --restart=unless-stopped -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama

echo "=========================================="
echo "Setup Complete! Ollama is running."
echo "Access it at: http://localhost:11434"
echo "=========================================="
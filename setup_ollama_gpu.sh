#!/bin/bash

# =======================================================================================
# Script Name:    setup_ollama_gpu.sh
# Description:    Automates the installation of Docker and the NVIDIA Container Toolkit 
#                 (pinned to v1.18.1), then launches Ollama with GPU support.
# Author:         Egor Alekseyevich Morozov
# Date:           2025-01-28
# Version:        1.0
# Requirements:   - Ubuntu/Debian-based Linux distribution
#                 - Sudo/Root privileges
#                 - NVIDIA GPU with proprietary drivers installed
# =======================================================================================


# ==========================================
# 1. Check and Install Docker
# ==========================================
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

# Update package list and install dependencies
sudo apt-get update && sudo apt-get install -y --no-install-recommends \
   curl \
   gnupg2

# Setup NVIDIA GPG Key and Repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Enable experimental packages support
sudo sed -i -e '/experimental/ s/^#//g' /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update

# Set specific version variable
export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.18.1-1

# Install specific version packages
# (Note: The redundant versionless install command has been removed to prevent accidental upgrades)
sudo apt-get install -y \
    nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
    nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
    libnvidia-container-tools=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
    libnvidia-container1=${NVIDIA_CONTAINER_TOOLKIT_VERSION}

# ==========================================
# 3. Configure Docker Runtime and Restart
# ==========================================
sudo nvidia-ctk runtime configure --runtime=docker

sudo systemctl restart docker

echo "Installation and configuration complete!"

# ==========================================
# 4. Verification Step
# ==========================================

echo "Verifying NVIDIA Container Toolkit installation with a test container..."
sudo docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi

# ==========================================
# 5. Run Ollama inside a Docker container
# ==========================================

echo "Running Ollama inside a Docker container..."

# Set container to restart unless stopped and map necessary volumes and ports
docker run -d --gpus=all --restart=unless-stopped -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
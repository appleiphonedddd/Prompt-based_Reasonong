# 🚀 Prompt-based Reasoning

> **A framework for evaluating Large Language Models (LLMs) on complex reasoning tasks using various prompting strategies**

## Contents

- [🚀 Prompt-based Reasoning](#-prompt-based-reasoning)
  - [Contents](#contents)
  - [Overview](#overview)
  - [Prerequisites](#prerequisites)
  - [Deployment](#deployment)
  - [Installation](#installation)
    - [1. System Setup](#1-system-setup)
    - [2. Setting up API Keys](#2-setting-up-api-keys)
    - [3. Deploying LLM](#3-deploying-llm)
  - [Evaluation](#evaluation)

## Overview

This project provides tools to benchmark reasoning capabilities of both local (Docker + vLLM) and cloud-based (Gemini, OpenAI) models. It supports various datasets (e.g., Game of 24) and prompting baselines

## Prerequisites

Before running the deployment scripts, ensure your system meets the following hardware and software requirements:

- **Operating System**: Ubuntu 24.04.03 LTS (Linux-based)
- **GPU**: NVIDIA GeForce RTX 5070 or higher (CUDA-enabled)
- **CUDA Toolkit**: 12.x (ensure compatibility with your GPU drivers)
- **Docker**: Required for running local models via vLLM
- **NVIDIA Docker Runtime**: For GPU support in containers


## Deployment

Create a virtual environment and install the Python libraries

```sh
conda env create -f env.yaml
conda activate Prompt
```

## Installation

### 1. System Setup (GPU Support)

**Option A: Automated (Recommended)**

```sh
./setup_ollama_gpu.sh
```

**Option B: Manual Installation**

```sh
# Add NVIDIA repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install NVIDIA Docker
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Verify GPU access
sudo docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi
```

### 2. Deploying Local Models

Start vLLM container with your chosen model:

```sh
# Qwen 2.5 7B
docker run -d --name qwen-vllm --gpus all -p 8000:8000 \
  vllm/vllm-openai:latest --model Qwen/Qwen2.5-7B-Instruct

# Llama 3.1 8B
docker run -d --name llama-vllm --gpus all -p 8000:8000 \
  vllm/vllm-openai:latest --model meta-llama/Llama-3.1-8B-Instruct

# Gemma 7B
docker run -d --name gemma-vllm --gpus all -p 8000:8000 \
  vllm/vllm-openai:latest --model google/gemma-7b-it
```

### 3. API Key Configuration (Optional)

API keys are **not required** for local Docker models. If needed for cloud APIs:

```sh
export API_KEY="your_openai_api_key"
export GEMINI_API_KEY="your_gemini_api_key"
```

## Evaluation

```sh
python main.py --model qwen2.5:7b --baseline standard --benchmark mgsm
```
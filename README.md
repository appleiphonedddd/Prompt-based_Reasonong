# 🚀 Prompt-based Reasoning

> **A framework for evaluating Large Language Models (LLMs) on complex reasoning tasks using Chain-of-Thought (CoT) and various prompting strategies**

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

This project provides tools to benchmark reasoning capabilities of both local (Ollama) and cloud-based (Gemini, OpenAI) models. It supports various datasets (e.g., Game of 24) and prompting baselines

## Prerequisites

Before running the deployment scripts, ensure your system meets the following hardware and software requirements:

- **Operating System**: Ubuntu 24.04.03 LTS (Linux-based)
- **GPU**: NVIDIA GeForce RTX 5070 or higher (CUDA-enabled)
- **CUDA Toolkit**: 12.x (ensure compatibility with your GPU driver)
- **Docker**: Required for running local Ollama instances


## Deployment

Create a virtual environment and install the Python libraries

```sh
conda env create -f env.yaml
conda activate Prompt
```

## Installation

### 1. System Setup
If you intend to use local models, run the setup script to configure Docker, the NVIDIA Container Toolkit, and Ollama

```sh
./setup_ollama_gpu.sh
```

### 2. Setting up API Keys

```sh
export API_KEY="your_api_key"
```

### 3. Deploying LLM

Reference [Ollama](https://ollama.com/search) for available more models

```sh
docker exec -it ollama ollama run qwen2:0.5b
```

Delete a model:

```sh
docker exec ollama ollama rm qwen2:0.5b
```

## Evaluation

```sh
python main.py \
  --provider gemini \
  --model gemini-2.5-flash \
  --dataset game24 \
  --baseline zero_shot_cot \
  --shots 0 \
  --output results/test_run.json
```
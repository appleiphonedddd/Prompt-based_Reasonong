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

This project provides tools to benchmark reasoning capabilities of both local (Docker + Ollama) and cloud-based (Gemini, OpenAI) models. It supports various datasets (e.g., Game of 24) and prompting baselines

## Prerequisites

Before running the deployment scripts, ensure your system meets the following hardware and software requirements:

- **Operating System**: Ubuntu 24.04.03 LTS (Linux-based)
- **GPU**: NVIDIA GeForce RTX 5070 or higher (CUDA-enabled)
- **CUDA Toolkit**: 12.x (ensure compatibility with your GPU drivers)
- **Docker**: Required for running local models via Ollama
- **NVIDIA Docker Runtime**: For GPU support in containers


## Deployment

Create a virtual environment and install the Python libraries

```sh
conda env create -f env.yaml
conda activate Prompt
```

## Installation

### 1. System Setup (GPU Support)

**Install ollama**

```sh
./setup_ollama_gpu.sh
```

### 2. Deploying Local Models with Ollama

Set API_KEY for Ollama:

```sh
export API_KEY="ollama"
```

Pull your desired model:

```sh
docker exec -it ollama ollama run qwen2.5:3b
```

Delete model:

```sh
docker exec -it ollama ollama rm qwen2.5:3b
```

## Evaluation

Run benchmarks on local Ollama models:

```sh
python main.py --model qwen2.5:3b --baseline standard
```
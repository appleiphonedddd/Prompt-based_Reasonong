### 🚀 Prompt-based Reasoning

## Contents

- [Contents](#contents)
  - [Getting Started](#getting-started)
        - [Requirements](#requirements)
  - [Deployment](#deployment)
  - [Author](#author)

### Getting Started

###### Requirements

- **Operating System**: Ubuntu 24.04.03 LTS (Linux-based)
- **GPU**: NVIDIA GeForce RTX 5070 (or higher, CUDA-enabled)
- **CUDA Toolkit**: 12.x (compatible with your GPU driver)


### Deployment

1. Create a virtual environment and install the Python libraries

```sh
conda env create -f env.yaml
conda activate Prompt
```

2. Apply for API keys

- [GPT](https://platform.openai.com/api-keys)

- [Gemini](https://aistudio.google.com/api-keys)

- [DeepSeek](https://platform.deepseek.com/usage)


3. Set the Google API key or OpenAI API key as an environment variable

```sh
export GOOGLE_API_KEY="your_api_key_here"
export OPENAI_API_KEY="your_api_key_here"
```

4. Run evaluation

```sh
python main.py \
  --provider gemini \
  --model gemini-2.5-flash \
  --dataset game24 \
  --baseline zero_shot_cot \
  --shots 0 \
  --output results/test_run.json
```

### Author

611221201@gms.ndhu.edu.tw

Egor Alekseyevich Morozov

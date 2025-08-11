#!/bin/bash

# install torch
uv pip install torch==2.5.0 torchvision -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# install FA2 and diffusers
uv pip install packaging ninja -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple && uv pip install flash-attn==2.7.0.post2 --no-build-isolation -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

uv pip install -r requirements-lint.txt -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# install fastvideo
uv pip install -e . -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

uv pip install ml-collections absl-py inflect==6.0.4 pydantic==1.10.9 huggingface_hub==0.24.0 protobuf==3.20.0 -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

uv pip install accelerate -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

uv pip install trl -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

uv pip install wandb -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

uv pip install pydantic==2.11.5 -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

uv pip install liger_kernel -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

uv pip install opencv-python -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

uv pip install image-reward -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

uv pip install git+https://github.com/openai/CLIP.git

uv pip install torchmetrics -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

uv pip install timm==1.0.13 -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

uv pip install peft -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

uv pip install diffusers==0.32.2 -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

uv pip install open-clip-torch -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

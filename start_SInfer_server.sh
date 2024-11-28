#!/bin/bash

python -m sinfer.entrypoints.openai.api_server \
    --model ./models/Meta-Llama-3-8B-Instruct \
    --dtype float16 \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.9 \
    --swap-space 64 \
    --api-key token-abc123 \
    # --kv-cache-dtype \
    # --device cuda \
    # --enable-prefix-caching \
    # --block-size 16 \ 目前mlu不支持
    # --host \
    # --port \
    # --uvicorn-log-level \
    # --lora-modules \
    # --chat-template \
    # --response-role \
    # --pipeline-parallel-size \
    # --max-num-batched-tokens \
    # --max-num-seqs \
    # --max-batch_size-to-capture \
    # --max-model-len \ 模型上下文长度
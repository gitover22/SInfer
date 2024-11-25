#!/bin/bash

python -m sinfer.entrypoints.openai.api_server \
    --model ./models/Meta-Llama-3-8B-Instruct \
    --dtype float16 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --swap-space 4 \
    --api-key token-abc123 \
    # --device cuda
    # --enable-prefix-caching \
    # --block-size 16 \
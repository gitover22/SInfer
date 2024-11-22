#!/bin/bash

python -m sinfer.entrypoints.openai.api_server \
    --model /home/zouguoqiang/SInfer/models/Meta-Llama-3-8B-Instruct \
    --dtype float16 \
    --api-key token-abc123
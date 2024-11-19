#!/bin/bash
set -ex

export TEST_MODEL_PATH="/data/AE/llm/models/Llama-2-7b-hf"
export HOST_IP="0.0.0.0"
export PORT="3000"

export TinyLlama_1_1B_Chat_v1_0_marlin="/data/sinfer/SInfer_tests_hf_models/TinyLlama-1.1B-Chat-v1.0-marlin"
export TinyLlama_1_1B_Chat_v1_0_GPTQ_4bit="/data/sinfer/SInfer_tests_hf_models/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit"
export TinyLlama_1_1B_Chat_v1_0_GPTQ_Marlin_4bit="/data/sinfer/SInfer_tests_hf_models/TinyLlama-1.1B-Chat-v1.0-GPTQ-Marlin-4bit"
export Llama_2_7B_Chat_GPTQ="/data/sinfer/SInfer_tests_hf_models/Llama-2-7B-Chat-GPTQ"
export OpenHermes_2_5_Mistral_7B_AWQ="/data/sinfer/SInfer_tests_hf_models/OpenHermes-2.5-Mistral-7B-AWQ"
export zephyr_7b_beta="/data/sinfer/SInfer_tests_hf_models/zephyr-7b-beta"
export gpt2="/data/sinfer/SInfer_tests_hf_models/gpt2"

RUN_ALL_TESTCASE() {
    ROOT_DIR=$PWD

    cd $ROOT_DIR/async_engine
    pytest

    cd $ROOT_DIR/core
    pytest

    cd $ROOT_DIR/distributed
    pytest test_comm_ops.py

    cd $ROOT_DIR/engine
    pytest test_computed_prefix_blocks.py
    pytest test_detokenization.py
    pytest test_multiproc_workers.py
    pytest test_skip_tokenizer_init.py
    pytest test_stop_reason.py
    pytest output_processor/test_multi_step.py

    cd $ROOT_DIR/entrypoints
    pytest test_guided_processors.py
    pytest test_llm_generate.py
    pytest test_server_oot_registration.py
    pytest openai/test_serving_chat.py

    cd $ROOT_DIR/kernels
    pytest test_activation.py
    pytest test_layernorm.py
    pytest test_feed_forward.py
    pytest bt_torch_ops

    cd $ROOT_DIR/model_executor
    pytest weight_utils.py

    cd $ROOT_DIR/models
    pytest test_oot_registration.py

    cd $ROOT_DIR/prefix_caching
    pytest test_prefix_caching.py

    cd $ROOT_DIR/quantization
    pytest test_configs.py

    cd $ROOT_DIR/samplers
    pytest test_ignore_eos.py
    pytest test_ranks.py

    cd $ROOT_DIR/tokenization
    pytest test_cached_tokenizer.py
    pytest test_detokenize.py
    pytest test_tokenizer_group.py

    cd $ROOT_DIR/worker
    pytest test_model_runner.py
    pytest test_swap.py

    cd $ROOT_DIR
    pytest test_config.py
    pytest test_logger.py
    pytest test_logits_processor.py
    pytest test_regression.py
    pytest test_sampling_params.py
    pytest test_sequence.py

    cd $ROOT_DIR/context_parallel
    export USE_CONTEXT_PARALLEL=True
    pytest test_context_parallel.py
    unset USE_CONTEXT_PARALLEL
}

RUN_ALL_TESTCASE 2>&1 | tee sinfer_test_results.log


### 简介

该example是SInfer中进行Context Parallel和Ring Attention的实验，mlu_hijack是对仓库代码的劫持，避免修改主仓库代码

### 支持模型

目前仅对LLaMA2系列模型进行了精度验证

### 支持板卡

暂不支持300系列设备

### 运行demo
```python
python examples/context_parallel/offline_inference.py
```

### 使用Context Parallel特性

设置环境变量export USE_CONTEXT_PARALLEL=1|True|true|TRUE， LLM主接口传入context_parallel_size参数

### 实现细节

- 为了使Ring Attention实现负载均衡，数据使用了zigzag的拆分方式
- 需要的MLU卡数为world_size = context_parallel_size * tensor_parallel_size，先拆cp， 然后拆tp
- 目前只是用作实验验证，context阶段采用cp，decoder阶段只在一个cp group上进行

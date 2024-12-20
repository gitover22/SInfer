### 1. 非page模式max_num_seqs自动调优工具

对于MLU370X8平台，在unpage模式下，可以通过调整`max_num_seqs`来提升性能。`tune_max_num_seqs.py`通过自动调参来搜索最佳`max_num_seqs`值。
- 用法示例
搜索固定配置下，使吞吐量最大`max_num_seqs`值，其中参数部分保持与`benchmark_latency.py`/`benchmark_throughput.py`一致。
```bash
python tools/utils/tune_max_num_seqs.py --backend sinfer --input-len 1024 --output-len 1024 --model /Path/to/Llama-2-70b-chat-hf/ -tp 1 --max-model-len 4096 --dtype float16 --num-prompts 10
```
通过执行上述命令，可以搜索得到最优`max_num_seqs`配置，在构建LLM对象时，作为参数传入使用。



### 2. SInfer调度分析辅助工具

#### 2.1 环境变量
设置环境变量开启调度分析: export SINFER_SCHEDULER_PROFILE_ENABLE=true

#### 2.2 离线测试
对于离线测试，测试结束后，会自动保存数据并打印出当前已经运行请求的信息

#### 2.3 在线测试
对于在线测试，获取调度数据的步骤如下：
1. 启动server
2. 运行client端测试
3. 等待client测试结束后，立即运行：python3 tools/utils/pull_server_scheduler_view.py，请求server端将数据保存下来
4. server端会打印出当前已经运行请求的信息
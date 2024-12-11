# from huggingface_hub import snapshot_download

# sql_lora_path = snapshot_download(repo_id="yard1/llama-2-7b-sql-lora-test") # Lora适配器模型

chinese_lora_path = "/home/zouguoqiang/SInfer/models/llama-3-chinese-8b-lora"
from sinfer import LLM, SamplingParams
from sinfer.lora.request import LoRARequest

llm = LLM(model="/home/zouguoqiang/SInfer/models/Meta-Llama-3-8B-Instruct", enable_lora=True,trust_remote_code=True,dtype="float16",tensor_parallel_size =1,gpu_memory_utilization=0.9) # 基础模型

sampling_params = SamplingParams(
    temperature=0.8, 
    top_p=0.95,
    max_tokens=500
)

prompts = [
     "请写一首诗,表达恋人之间的思念之情",
     "请根据中国人的起名习惯,为一个姓庞的新生儿起名",
]

outputs = llm.generate(
    prompts,
    sampling_params,
    lora_request=LoRARequest("chinese_lora_adapter", 1, chinese_lora_path)
)


for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    print("\n===========================================================================\n")
    sum += len(output.outputs[0].token_ids)
    print("\n===========================================================================\n")
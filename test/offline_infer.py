from sinfer import LLM, SamplingParams
import time
from tool import read_prompts_from_csv
sampling_params = SamplingParams(
    temperature=0.8, 
    top_p=0.95,
    max_tokens=256
)
# print(sampling_params) 

def main(args):
    # 实例化LLM对象
    prompts = read_prompts_from_csv("prompts.csv",args.num)
    llm = LLM(model="./Meta-Llama-3-8B-Instruct",trust_remote_code=True,dtype="float16",tensor_parallel_size =2,gpu_memory_utilization=1)

    s = time.time()
    outputs = llm.generate(prompts, sampling_params)
    e = time.time()
    sum:int =0
    # 输出
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
        print("\n===========================================================================\n")
        sum += len(output.outputs[0].token_ids)
        print("\n===========================================================================\n")
    total_time = e - s
    print("total time: "+str(total_time))
    tokens_per_second = sum / total_time
    print(f"Total tokens generated: {sum}")
    print(f"prompts per seconds: {len(prompts)/total_time:.2f}")
    print(f"Tokens per second: {tokens_per_second:.2f}")
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='Demo on using the LLMRuntime class directly')
    parser.add_argument("--num", type=int, default=8)
    args = parser.parse_args()
    main(args)
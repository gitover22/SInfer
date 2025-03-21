from sinfer import LLM, SamplingParams
from tool import read_prompts_from_csv
sampling_params = SamplingParams(
    # temperature=0.8, 
    # top_p=0.95,
    max_tokens=128
)

def main(args):
    prompts = read_prompts_from_csv("prompts.csv",args.num)
    llm = LLM(model="./Meta-Llama-3-8B-Instruct",trust_remote_code=True,dtype="float16",tensor_parallel_size =4,gpu_memory_utilization=0.8)
    outputs = llm.generate(prompts, sampling_params)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description='Demo on using the LLMRuntime class directly')
    parser.add_argument("--num", type=int, default=8)
    args = parser.parse_args()
    main(args)

from sinfer import LLM, SamplingParams
import time
# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    # "The capital of France is",
    # "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(
    temperature=0.8, 
    top_p=0.95,
    max_tokens=500
)

# 实例化LLM对象
llm = LLM(model="./2b_sft_model",trust_remote_code=True,dtype="float16")
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
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
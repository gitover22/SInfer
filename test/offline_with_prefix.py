from sinfer import LLM, SamplingParams

prefix = (
    "You are an expert school principal, skilled in effectively managing "
    "faculty and staff. Draft 10-15 questions for a potential first grade "
    "Head Teacher for my K-12, all-girls', independent school that emphasizes "
    "community, joyful discovery, and life-long learning. The candidate is "
    "coming in for a first-round panel interview for a 8th grade Math "
    "teaching role. They have 5 years of previous teaching experience "
    "as an assistant teacher at a co-ed, public school with experience "
    "in middle school math teaching. Based on these information, fulfill "
    "the following paragraph: ")

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.0)

llm = LLM(model="/home/zouguoqiang/SInfer/test/Meta-Llama-3-8B-Instruct", enable_prefix_caching=True,dtype="float16",tensor_parallel_size =1,gpu_memory_utilization=0.9)

generating_prompts = [prefix + prompt for prompt in prompts]

outputs = llm.generate(generating_prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

print("-" * 80)

# llm.generate 调用会将所有的提示（prompts）进行批处理，并在资源允许的情况下一次性发送整个批次。
# 前缀（prefix）只有在第一个批次处理完成后才会被缓存，因此我们需要先调用一次 generate 来计算前缀并将其缓存。
outputs = llm.generate(generating_prompts[0], sampling_params)

# 后续批次可以利用缓存的前缀
outputs = llm.generate(generating_prompts, sampling_params)

# Print the outputs. You should see the same outputs as before
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

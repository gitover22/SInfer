from openai import OpenAI

openai_api_key = "token-abc123"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id

stream = False # 流输出
# @app.post("/v1/completions")
completion = client.completions.create(
    model=model,
    prompt="write some python program",
    echo=False,
    n=1,
    stream=stream,
    logprobs=3)

print("Completion results:")
if stream:
    for c in completion:
        # 只打印每个返回结果中的文本内容
        print(c.choices[0].text,end='')  # 获取每个结果中的文本
else:
    # 如果是非流式输出，打印文本部分
    print(completion.choices[0].text)
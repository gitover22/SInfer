from openai import OpenAI

# 初始化 OpenAI 客户端
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123",
)

# 定义用户输入
user_message = [{
        "role": "system",
        "content": "You are a helpful assistant."
    }, {
        "role": "user",
        "content": "Who won the world series in 2020?"
    }, {
        "role":"assistant",
        "content":"The Los Angeles Dodgers won the World Series in 2020."
    }, {
        "role": "user",
        "content": "Where was it played?"
    }]
# 请求模型生成结果，该方法对应post请求
completion = client.chat.completions.create(
    model="/home/zouguoqiang/SInfer/models/Meta-Llama-3-8B-Instruct",
    messages=user_message,
    # max_tokens=1000  # 确保生成内容不超过上下文窗口限制
)
# client.models()
print(client.models)
# 打印用户输入和模型生成内容
print("="*50)
print("User Prompt:", user_message[3]['content'])
print("="*50)
print("Model Response:", completion.choices[0].message.content)
print("="*50)
print("\n\n")

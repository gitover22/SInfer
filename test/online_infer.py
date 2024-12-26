from openai import OpenAI

# 初始化 OpenAI 客户端
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123",
)

# 定义用户输入
user_message1 = [{
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
user_message2 = [{
        "role": "user",
        "content": "Where was it played?"
    }]
# 请求模型生成结果，该方法对应post请求
completion = client.chat.completions.create(
    model="./test/Meta-Llama-3-8B-Instruct",
    messages=user_message2,
    max_tokens=256  # 限制模型最大生成内容长度，注意token_num(messages长度) + max_tokens < max_model_len(模型支持的最大上下文长度)
)
# client.models()
print(client.models)
# 打印用户输入和模型生成内容
print("="*50)
print("User Prompt:", user_message2[0]['content'])
print("="*50)
print("Model Response:", completion.choices[0].message.content)
print("="*50)
print("\n\n")

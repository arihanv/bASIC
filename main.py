from openai import OpenAI

client = OpenAI(
    base_url="https://api--openai-vllm--d8zwcx9rqzwl.code.run/v1",
    api_key="EMPTY"
)

completion = client.chat.completions.create(
  model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
  stream=True,
  messages=[
    {"role": "user", "content": "What is the capital of France?"}
  ]
)

for chunk in completion:
    print(chunk.choices[0].delta.content, end="", flush=True)
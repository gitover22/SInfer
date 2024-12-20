import argparse
import requests

""" 向服务器发送POST请求, 保存调度器剖析数据"""
def post_http_request(api_url: str) -> requests.Response:
    headers = {"User-Agent": "Test Client"}
    pload = {
        "model": "save_scheduler_view",
        "prompt": "",
        "n": 1,
        "temperature": 0.0,
        "max_tokens": 16,
        "stream": True,
    }
    response = requests.post(api_url, headers=headers, json=pload, stream=True)
    return response


if __name__ == "__main__":
    api_url = f"http://localhost:8000/v1/completions"
    _ = post_http_request(api_url)

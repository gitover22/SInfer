import argparse
import requests

""" Post a request to server, let server save scheduler profiling data. """
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=6000)
    args = parser.parse_args()
    api_url = f"http://{args.host}:{args.port}/v1/completions"

    _ = post_http_request(api_url)

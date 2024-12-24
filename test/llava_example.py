import argparse
import os
import subprocess

import torch

from sinfer import LLM
from sinfer.task import MultiModalData


def run_llava_pixel_values():
    llm = LLM(
        model="./llava-1.5-7b-hf/",
        image_input_type="pixel_values",
        image_token_id=32000,
        image_input_shape="1,3,336,336",
        image_feature_size=576,
        dtype="float16",
    )

    prompt = "<image>" * 576 + (
        "\nUSER: What is the content of this image?\nASSISTANT:")

    images = torch.load("images/stop_sign_pixel_values.pt")

    outputs = llm.generate(prompt,
                           multi_modal_data=MultiModalData(
                               type=MultiModalData.Type.IMAGE, data=images))
    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)


def run_llava_image_features():
    llm = LLM(
        model="./llava-1.5-7b-hf/",
        image_input_type="image_features",
        image_token_id=32000,
        image_input_shape="1,576,1024",
        image_feature_size=576,
        dtype="float16",
    )

    prompt = "<image>" * 576 + (
        "\nUSER: What is the content of this image?\nASSISTANT:")

    images = torch.load("images/stop_sign_image_features.pt") # 应该由另一个在线或离线组件提供

    outputs = llm.generate(prompt,
                           multi_modal_data=MultiModalData(
                               type=MultiModalData.Type.IMAGE, data=images))
    for o in outputs:
        generated_text = o.outputs[0].text
        print(generated_text)


def main(args):
    if args.type == "pixel_values":
        run_llava_pixel_values()
    else:
        run_llava_image_features()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo on Llava")
    parser.add_argument("--type",
                        type=str,
                        choices=["pixel_values", "image_features"],
                        default="pixel_values",
                        help="image input type")
    args = parser.parse_args()
    # Download from s3
    # s3_bucket_path = "s3://air-example-data-2/vllm_opensource_llava/"
    # local_directory = "images"

    # os.makedirs(local_directory, exist_ok=True)

    # # Use AWS CLI to sync the directory, assume anonymous access
    # subprocess.check_call([
    #     "aws",
    #     "s3",
    #     "sync",
    #     s3_bucket_path,
    #     local_directory,
    #     "--no-sign-request",
    # ])
    main(args)

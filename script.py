#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import torch
import os
import signal
import time


def parse_args():
    parser = argparse.ArgumentParser(
        description="Chat with huihui-ai/Huihui-Qwen3.8-27B-abliterated."
    )

    parser.add_argument(
        "--base_model",
        type=str,
        default="huihui-ai/Huihui-Qwen3.8-27B-abliterated",
        help="HuggingFace repo or local path of the base model.",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Data type for loading the model.",
    )

    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="Device map for model loading.",
    )

    return parser.parse_args()


def main():
    # Максимальное время работы:
    # 20700 секунд = 5 часов 45 минут
    MAX_RUNTIME = int(os.getenv("MAX_RUNTIME_SECONDS", "20700"))
    START_TIME = time.time()

    cpu_count = os.cpu_count() or 2
    half_cpu_count = max(1, cpu_count // 2)

    print(f"Number of CPU cores in the system: {cpu_count}")

    os.environ["MKL_NUM_THREADS"] = str(half_cpu_count)
    os.environ["OMP_NUM_THREADS"] = str(half_cpu_count)

    torch.set_num_threads(half_cpu_count)

    print(f"PyTorch threads: {torch.get_num_threads()}")
    print(f"MKL threads: {os.getenv('MKL_NUM_THREADS')}")
    print(f"OMP threads: {os.getenv('OMP_NUM_THREADS')}")

    args = parse_args()

    print(f"Load Model {args.base_model} ...")

    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]

    # Загрузка модели
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch_dtype,
        device_map=args.device_map,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    # Загрузка токенизатора
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True
    )

    messages = []

    class CustomTextStreamer(TextStreamer):

        def __init__(
            self,
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        ):
            super().__init__(
                tokenizer,
                skip_prompt=skip_prompt,
                skip_special_tokens=skip_special_tokens
            )

            self.generated_text = ""
            self.stop_flag = False

            self.init_time = time.time()
            self.end_time = None
            self.first_token_time = None

            self.think_tokens_count = 0
            self.token_count = 0

        def on_finalized_text(
            self,
            text: str,
            stream_end: bool = False
        ):
            if self.first_token_time is None and text.strip():
                self.first_token_time = time.time()

            if stream_end:
                self.end_time = time.time()

            self.generated_text += text

            tokens = self.tokenizer.encode(
                text,
                add_special_tokens=False
            )

            self.token_count += len(tokens)

            if (
                self.think_tokens_count == 0
                and "</think>" in self.generated_text
            ):
                self.think_tokens_count = self.token_count

            print(text, end="", flush=True)

            if self.stop_flag:
                raise StopIteration

        def stop_generation(self):
            self.stop_flag = True
            self.end_time = time.time()

        def get_metrics(self):
            if self.end_time is None:
                self.end_time = time.time()

            total_time = self.end_time - self.init_time

            tokens_per_second = (
                self.token_count / total_time
                if total_time > 0
                else 0
            )

            first_token_latency = (
                self.first_token_time - self.init_time
                if self.first_token_time is not None
                else None
            )

            return {
                "init_time": self.init_time,
                "first_token_time": self.first_token_time,
                "first_token_latency": first_token_latency,
                "end_time": self.end_time,
                "total_time": total_time,
                "total_tokens": self.token_count,
                "think_tokens_count": self.think_tokens_count,
                "real_tokens_count": (
                    self.token_count - self.think_tokens_count
                ),
                "tokens_per_second": tokens_per_second,
            }

    def generate_stream(
        model,
        tokenizer,
        messages,
        enable_thinking,
        skip_prompt,
        skip_special_tokens,
        max_new_tokens
    ):
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking
        )

        inputs = tokenizer(
            text,
            return_tensors="pt",
        ).to(model.device)

        streamer = CustomTextStreamer(
            tokenizer,
            skip_prompt=skip_prompt,
            skip_special_tokens=skip_special_tokens
        )

        def signal_handler(sig, frame):
            streamer.stop_generation()
            print(
                "\n[Generation stopped by user with Ctrl+C]"
            )

        signal.signal(
            signal.SIGINT,
            signal_handler
        )

        print("Response: ", end="", flush=True)

        try:
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                streamer=streamer
            )

            del generated_ids

        except StopIteration:
            print("\n[Stopped by user]")

        del inputs

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        signal.signal(
            signal.SIGINT,
            signal.SIG_DFL
        )

        return (
            streamer.generated_text,
            streamer.stop_flag,
            streamer.get_metrics()
        )

    skip_prompt = True
    skip_special_tokens = True
    enable_thinking = False

    while True:

        # Автоматическое завершение через 5 часов 45 минут
        if time.time() - START_TIME >= MAX_RUNTIME:
            print(
                "\nMaximum runtime reached. Exiting..."
            )
            break

        print(
            f"skip_prompt = {skip_prompt}."
        )

        print(
            f"skip_special_tokens = {skip_special_tokens}."
        )

        print(
            f"enable_thinking = {enable_thinking}."
        )

        user_input = input("User: ").strip()

        if user_input.lower() == "/exit":
            print("Exiting chat.")
            break

        if user_input.lower() == "/clear":
            messages = []
            print(
                "Chat history cleared. "
                "Starting a new conversation."
            )
            continue

        if user_input.lower() == "/skip_prompt":
            skip_prompt = not skip_prompt
            continue

        if user_input.lower() == "/skip_special_tokens":
            skip_special_tokens = not skip_special_tokens
            continue

        if user_input.lower() == "/enable_thinking":
            enable_thinking = not enable_thinking
            continue

        if not user_input:
            print(
                "Input cannot be empty. "
                "Please enter something."
            )
            continue

        messages.append({
            "role": "user",
            "content": user_input
        })

        response, stop_flag, metrics = generate_stream(
            model,
            tokenizer,
            messages,
            enable_thinking,
            skip_prompt,
            skip_special_tokens,
            40960
        )

        print("\n\nMetrics:")

        for key, value in metrics.items():
            print(f"  {key}: {value}")

        print("", flush=True)

        if stop_flag:
            continue

        messages.append({
            "role": "assistant",
            "content": response
        })


if __name__ == "__main__":
    main()

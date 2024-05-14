import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vllm import LLM, SamplingParams

import argparse
from numpy import random
import json

def load_vllm_model(model_path):
    llm = LLM(model=model_path)
    return llm

def generate_prompt_landmark(n_garbage, seed, percent):
    """Generates a text file and inserts a passkey at a random position."""
    rnd_state = random.get_state()
    random.seed(seed)
    n_garbage_prefix = int(percent * n_garbage)
    n_garbage_suffix = n_garbage - n_garbage_prefix

    # Task description and garbage text in Mandarin (Taiwan)
    task_description = "有一個重要的信息隱藏在大量無關的文字中。找到它並記住它。我會考你這裡面的重要信息。"
    garbage = "草是綠的。天空是藍的。太陽是黃色的。我們開始吧。來來回回。"
    garbage_inf = " ".join([garbage] * 50000)
    assert len(garbage_inf) >= n_garbage
    garbage_prefix = garbage_inf[:n_garbage_prefix]
    garbage_suffix = garbage_inf[:n_garbage_suffix]
    pass_key = random.randint(50000, 500000)

    # Information line and final question in Mandarin (Taiwan)
    information_line = f"通行碼是 {pass_key}。記住它。{pass_key} 是通行碼。"
    final_question = "通行碼是什麼？通行碼是"
    lines = [
        task_description,
        garbage_prefix,
        information_line,
        garbage_suffix,
        final_question,
    ]
    random.set_state(rnd_state)
    return "\n".join(lines), str(pass_key)

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--model', type=str, default="mistral")
    parser.add_argument('--pretraining_length', type=int, default=32000)
    parser.add_argument('--scale', type=str, default="13b")
    parser.add_argument('--max_length', type=str, default="256k")
    parser.add_argument('--min_length', type=str, default="1k")
    parser.add_argument('--gap', type=str, default="8k")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--num_tests', type=int, default=10, help='number of repeat testing for each length')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_config()

    output_name = f"output.jsonl"
    print("results will be saved to:", output_name)
    model_path = args.model
    model = load_vllm_model(model_path)

    # Hyperparameters
    k = 1000
    max_length = int(args.max_length.replace("k", '')) * k
    min_length = int(args.min_length.replace("k", '')) * k
    gap = int(args.gap.replace("k", '')) * k
    num_per = 10
    depth_percent = 1 / num_per

    length_list = [i for i in range(min_length, max_length + 1, gap)]

    results = []
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        use_beam_search=True,
        best_of=4,
        max_tokens=5,
    )
    for length in length_list:
        # This is a rough ratio to control the number of texts and tokens
        n_garbage = int(3.75 * length // k * k)

        depths = [depth_percent * i for i in range(1, num_per + 1)]
        for depth in depths:
            passed_tests = 0
            all_accuries = {}
            prompts = []
            answers = []
            for j in range(args.num_tests):
                prompt, answer = generate_prompt_landmark(n_garbage, j, depth)
                prompts.append(prompt)
                answers.append(answer)

            outputs = model.generate(prompts, sampling_params)
            for output, answer in zip(outputs, answers):
                print("[prediction]:  ", repr(output.outputs[0].text))
                print("[ground truth]:  ", repr(answer))
                if answer in output.outputs[0].text:
                    passed_tests += 1
            accuracy = float(passed_tests) / args.num_tests
            res = {"context_length": f"{length // k}k", "depth_percent": depth * 100, "score": accuracy}
            results.append(res)
            print(res)
            with open(output_name, "a") as f:
                print(json.dumps(res), file=f)

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vllm import LLM, SamplingParams

import argparse
from numpy import random
import json
import torch

gpu_count = torch.cuda.device_count()

def load_vllm_model(model_path, max_length=1024, tensor_parallel_size=gpu_count):
    llm = LLM(model=model_path, tensor_parallel_size=tensor_parallel_size, max_model_len=max_length)
    return llm

def generate_prompt_landmark(n_garbage, seed, percent):
    """Generates a text file and inserts a passkey at a random position."""
    rnd_state = random.get_state()
    random.seed(seed)
    n_garbage_prefix = int(percent * n_garbage)
    n_garbage_suffix = n_garbage - n_garbage_prefix

    # Task description and garbage text in Mandarin (Taiwan)
    task_description = "有一個重要的訊息隱藏在《出師表》的文字中。找到它並記住它。我會考你這裡面的重要訊息。"
    garbage = """臣亮言：先帝創業未半而中道崩殂，今天下三分，益州疲弊，此誠危急存亡之秋也。然侍衞之臣不懈於內，忠志之士忘身於外者，蓋追先帝之殊遇，欲報之於陛下也。

誠宜開張聖聽，以光先帝遺德，恢弘志士之氣，不宜妄自菲薄，引喻失義，以塞忠諫之路也。

宮中府中，俱爲一體，陟罰臧否，不宜異同。若有作奸犯科及爲忠善者，宜付有司論其刑賞，以昭陛下平明之治，不宜偏私，使內外異法也。

侍中、侍郎郭攸之、費禕、董允等，此皆良實，志慮忠純，是以先帝簡拔以遺陛下。愚以爲宮中之事，事無大小，悉以咨之，然後施行，必能裨補闕漏，有所廣益。

將軍向寵，性行淑均，曉暢軍事，試用之於昔日，先帝稱之曰能，是以衆議舉寵爲督。愚以爲營中之事，悉以咨之，必能使行陣和穆，優劣得所也。

親賢臣，遠小人，此先漢所以興隆也；親小人，遠賢臣，此後漢所以傾頹也。先帝在時，每與臣論此事，未嘗不歎息痛恨於桓、靈也。侍中、尚書、長史、參軍，此悉貞亮死節之臣也，願陛下親之信之，則漢室之隆，可計日而待也。

臣本布衣，躬耕於南陽，苟全性命於亂世，不求聞達於諸侯。先帝不以臣卑鄙，猥自枉屈，三顧臣於草廬之中，諮臣以當世之事，由是感激，遂許先帝以驅馳。後值傾覆，受任於敗軍之際，奉命於危難之間，爾來二十有一年矣。先帝知臣謹慎，故臨崩寄臣以大事也。受命以來，夙夜憂歎，恐託付不效，以傷先帝之明，故五月渡瀘，深入不毛。今南方已定，兵甲已足，當獎率三軍，北定中原，庶竭駑鈍，攘除姦凶，興復漢室，還於舊都，此臣所以報先帝，而忠陛下之職分也。至於斟酌損益，進盡忠言，則攸之、禕、允之任也。

願陛下託臣以討賊興復之效；不效，則治臣之罪，以告先帝之靈。若無興德之言，則責攸之、禕、允等之慢，以彰其咎。陛下亦宜自課，以諮諏善道，察納雅言。深追先帝遺詔，臣不勝受恩感激。

今當遠離，臨表涕泣，不知所云。""".strip()
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
    parser.add_argument('--scale', type=str, default="8b")
    parser.add_argument('--max_length', type=str, default="128k")
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

    # Hyperparameters
    k = 1000
    max_length = int(args.max_length.replace("k", '')) * k
    min_length = int(args.min_length.replace("k", '')) * k
    gap = int(args.gap.replace("k", '')) * k
    num_per = 10
    depth_percent = 1 / num_per

    model = load_vllm_model(model_path, max_length=max_length, tensor_parallel_size=gpu_count)

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
        n_garbage = int(length / 1.05 // k * k)

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

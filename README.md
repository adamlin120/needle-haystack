Adapted form ChunkLlama's needle in haystack evals @ https://github.com/HKUNLP/ChunkLlama/tree/main/needle_in_a_haystack to use vLLM.

# Needle in a 出師表

This project aims to test the ability of language models to find important information hidden within large amounts of irrelevant text, specifically using the text from 《出師表》 by 諸葛亮. The script generates prompts by embedding a passkey within the text and then checks the language model's ability to recall this key.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [License](#license)

## Installation

To run this project, you'll need to install the required dependencies. You can install them using pip:

```bash
pip install vllm numpy argparse
```

Additionally, you'll need to have the appropriate language model from Hugging Face installed. For example:

```bash
pip install transformers
```

## Usage

The script `eval.py` is used to run the tests. You can configure various parameters such as the model, the length of the texts, and the number of tests to run.

### Command Line Arguments

- `--model`: The model name or path to be used (default: `mistral`).
- `--pretraining_length`: The length of pretraining data in tokens (default: `32000`).
- `--scale`: The scale or size of the model (default: `13b`).
- `--max_length`: The maximum context length for testing, specified in kilobytes (default: `256k`).
- `--min_length`: The minimum context length for testing, specified in kilobytes (default: `1k`).
- `--gap`: The gap between different context lengths during testing, specified in kilobytes (default: `8k`).
- `--gpu`: The GPU index to use for running the model (default: `0`).
- `--num_tests`: The number of repeat tests for each context length (default: `10`).

### Running the Script

To run the script with default parameters:

```bash
python eval.py --model gradientai/Llama-3-8B-Instruct-Gradient-1048k --max_length 256k --min_length 1k --gap 8k --gpu 0 --num_tests 10
```

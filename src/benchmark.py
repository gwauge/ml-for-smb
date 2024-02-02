#!/usr/bin/env python3
from statistics import median
import typing
import pandas as pd
import logging
import sys
import os

import requests

BENCHMARK_FOLDER = "../benchmarks"

logging.basicConfig(filename=os.path.join(BENCHMARK_FOLDER, "main.log"), encoding='utf-8', level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


PROMPTS = [
    "../prompts/long-doc-json-export.txt",
    # "../prompts/long-doc-csv-export.txt",
    # "../prompts/long-doc-yesno-export.txt",
]
RUNS_PER_MODEL= 5

def read_prompt(path: str):
    with open(path, 'r') as f:
        return f.read()

def mixtral(prompt: str):
    response = requests.post('http://localhost:11434/api/generate', json={
        'model': 'mixtral:8x7b-instruct-v0.1-q3_K_M',
        'prompt': prompt,
        "stream": False,
    })
    data = response.json()

    return {
        "tps": data['eval_count'] / (data['eval_duration'] / 1_000_000_000),
        "response": data["response"]
    }

def leolm(prompt):
    response = requests.post('http://localhost:8080/completion',
        headers={"Content-Type": "application/json"},
        json={"prompt": prompt}
    )
    data = response.json()

    return {
        "tps": data["timings"]["predicted_per_second"],
        "response": data["content"]
    }

MODELS: typing.List[typing.Tuple[str, typing.Callable[[str], object]]] = [
    # ("mixtral:8x7b-instruct-v0.1-q3_K_M", mixtral),
    # ("discolm-7b-q8_0", leolm),
    ("miqu-1-70b-q4_k_m", leolm),
]

def main():
    # import datafram from csv
    export = pd.read_csv(os.path.join(BENCHMARK_FOLDER, "benchmark.csv"))
    # export = pd.DataFrame(columns=["prompt", "model", "tps"])
    for prompt_file in PROMPTS:
        logging.info(f"[PROMPT] {prompt_file}")
        prompt = read_prompt(prompt_file)

        for model_name, model in MODELS:
            logging.info(f"[MODEL] {model_name}")

            results = []
            for _ in range(RUNS_PER_MODEL):
                data = model(prompt)
                # data = { "tps": 1, "response": "hello" } # mock data
                results.append(data['tps'])
                logging.debug(f"\t[RUN {_} TPS: {data['tps']:.2f} - Response: {data['response']}")

            result = median(results)
            logging.info(f"[RESULT] {result:.2f}")

            # add result to export dataframe
            export = pd.concat([export, pd.DataFrame([{
                "prompt": prompt_file,
                "model": model_name,
                "tps": result,
            }])], ignore_index=True)

    # export dataframe to csv
    csv_path = os.path.join(BENCHMARK_FOLDER, "benchmark.csv")
    logging.debug(f"Exporting to {csv_path}")
    export.to_csv(csv_path)

if __name__ == '__main__':
    main()

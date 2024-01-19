#!/opt/conda/bin/python
import requests

def main():
    # request ollama server with prompt from text file
    with open('../prompts/long-doc-json-export.txt', 'r') as f:
        prompt = f.read()
    response = requests.post('http://localhost:11434/api/generate', json={
        'model': 'mixtral:8x7b-instruct-v0.1-q3_K_M',
        'prompt': prompt,
        "stream": False,
    })
    data = response.json()
    tps = data['eval_count'] / (data['eval_duration'] / 1_000_000_000)
    print(f"Response: {data['response']}")
    print(f"Tokens/s: {tps:.2f}")

if __name__ == '__main__':
    main()

#!/usr/bin/env python3

# Use a pipeline as a high-level helper
from transformers import pipeline
import torch

def format_response(response):
    # parse the last response from the generator
    # and return the formatted version
    text = response[0]["generated_text"].split("<|im_start|>assistant\n")[-1].replace("<|im_end|>", "").strip()
    return text

system_prompt = """<|im_start|>system
Dies ist eine Unterhaltung zwischen einem intelligenten, hilfsbereitem KI-Assistenten und einem Nutzer.
Der Assistent gibt ausführliche, hilfreiche und ehrliche Antworten.<|im_end|>\n\n"""
prompt_format = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
prompt = "Wie sieht ein typischer Weihnachtsbaum in Deutschland aus?"

# generator = pipeline("text-generation", model="LeoLM/leo-hessianai-7b-chat")
generator = pipeline(model="LeoLM/leo-hessianai-7b-chat", device="cuda", torch_dtype=torch.float16, trust_remote_code=False) # True for flash-attn2 else False
# print(generator(prompt_format.format(prompt=prompt), do_sample=True, top_p=0.95, max_length=8192))

print("USER: " + prompt)
conversation = system_prompt + prompt_format.format(prompt=prompt)

while True:  # or some condition to end the conversation
    response = generator(conversation, do_sample=True, top_p=0.95, max_length=8192)
    # format_response is a function you need to implement to format the generator's response

    formatted = format_response(response)
    print("BOT:", formatted)
    next_prompt = input("USER: ")
    conversation += formatted + "<|im_end|>\n" + prompt_format.format(prompt=next_prompt)

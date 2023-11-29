#!/usr/bin/env python3
# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="LeoLM/leo-hessianai-13b-chat")
text = "Wie sieht ein typischer Weihnachtsbaum in Deutschland aus?"
result = pipe(text, max_length=8192, do_sample=True, top_p=0.95)

# create a chat interface
# while text != "":
#     result = pipe(text, max_length=8192, do_sample=True, top_p=0.95)
#     print("[BOT]: ", result[0]["generated_text"])
#     text = input("[USER]: ")




#!/usr/bin/env python3

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, Conversation
from optimum.onnxruntime import ORTModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("LeoLM/leo-hessianai-7b-chat")
# model = AutoModelForCausalLM.from_pretrained("LeoLM/leo-hessianai-7b-chat")

onnx_model = "leolm-7b-chat-onnx"
model = ORTModelForCausalLM.from_pretrained(onnx_model)
tokenizer = AutoTokenizer.from_pretrained(onnx_model)

generator = pipeline("conversational", model=model, tokenizer=tokenizer)
# pred = generator(question, context, do_sample=True, top_p=0.95, max_length=8192)

conversation = Conversation([
    { "role": "system",
    "content": "Dies ist eine Unterhaltung zwischen einem intelligenten, hilfsbereitem KI-Assistenten und einem Nutzer. Der Assistent gibt ausführliche, hilfreiche und ehrliche Antworten." },
    { "role": "user", "content": "Wie sieht ein typischer Weihnachtsbaum in Deutschland aus?"}
])
conversation = generator(conversation)
print(conversation.generated_responses[-1].strip())

while True:
    next_prompt = input("USER: ")
    conversation.add_user_input(next_prompt)
    conversation = generator(conversation)
    print(conversation.generated_responses[-1].strip())

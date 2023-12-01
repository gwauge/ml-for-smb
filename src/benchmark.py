#!/usr/bin/env python3
from time import time

from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, Conversation
from optimum.onnxruntime import ORTModelForCausalLM

onnx_model = "models/leolm-7b-chat-onnx"
models = [
    (
        AutoAWQForCausalLM.from_quantized("TheBloke/leo-hessianai-7B-chat-AWQ"),
        AutoTokenizer.from_pretrained("TheBloke/leo-hessianai-7B-chat-AWQ")
    ),
    (
        ORTModelForCausalLM.from_pretrained(onnx_model),
        AutoTokenizer.from_pretrained(onnx_model)
    ),
    (
        AutoModelForCausalLM.from_pretrained("LeoLM/leo-hessianai-7b-chat"),
        AutoTokenizer.from_pretrained("LeoLM/leo-hessianai-7b-chat")
    )
]

# run each model with different batch sizes and measure time

batch_sizes = [1, 2, 4]
for model, tokenizer in models:
    for batch_size in batch_sizes:
        # Generate a batch of inputs
        inputs = tokenizer("Hallo, wie geht es dir?", return_tensors='pt').input_ids
        inputs = inputs.repeat(batch_size, 1)

        # Start the timer
        start_time = time()

        # Generate responses
        model['model'].generate(inputs)

        # Stop the timer and print the time taken
        end_time = time()
        print(f"Model: {model['model']}, Batch size: {batch_size}, Time taken: {end_time - start_time} seconds")

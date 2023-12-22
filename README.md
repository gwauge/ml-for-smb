# Documentation

## Helpers
Execute in the background with 10min (600s) timeout and pipe output to `output.log`
```bash
nohup timeout 600 python src/leolm.py > output.log 2>&1 &
```
View output
```bash
tail -f output.log
```

## llama.cpp
### Manually download & convert model to GGUF format
Based on [Tutorial: How to convert HuggingFace model to GGUF format](https://github.com/ggerganov/llama.cpp/discussions/2948)

### Use already converted GGUF models
Remeber to adjust the model ID to your model
```bash
huggingface-cli download TheBloke/leo-hessianai-13B-GGUF leo-hessianai-13b.Q8_0.gguf --local-dir models/ --local-dir-use-symlinks False
```

### Run model using llama.cpp
```bash
llama.cpp/main -m models/leo-hessianai-7b.Q4_K_M.gguf -n -2 -c 0 -ngl 33 -p "Das Wetter in Potsdam soll"
```
```bash
llama.cpp/main -ngl 41 -m models/leo-hessianai-13b-chat.Q8_0.gguf --color -c 0 --temp 0.7 --repeat_penalty 1.1 -n -1 -f prompts/long-doc-csv-export.txt
```
```bash
llama.cpp/main -m models/leo-hessianai-7b-chat.Q8_0.gguf -n -1 -c 0 -ngl 33 -i -r "Benutzer:" -f prompts/rede-mit-bob.txt
```

#### Parameters
- `-m` path to the model
- `-n` number of tokens to generate, `-2` means generate until context is full
- `-c` context size, `0` means  load from model (LeoLM uses `8192`)
- `-ngl` n GPU layers, offload tensor layers of gguf model to GPU
    - use `0` to run on CPU only
- `-p` prompt
- `--batch-size` can be used, however it does not seem to make a difference in inference time, LeoLM default is `512` (I believe)

### Run model using HTTP server (powered by llama.cpp)
[Full documentation](https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md)
```bash
llama.cpp/server -m models/leo-hessianai-13b-chat.Q8_0.gguf -ngl 41
```

## ONNX
### Convert model to ONNX
```bash
optimum-cli export onnx --model LeoLM/leo-hessianai-7b-chat models/leolm-7b-chat-onnx/
```

## Pitfalls
### llama.cpp
- has to be compiled with LLAMA_cuBLAS=1 to enable GPU support
    - might have to update the Makefile to correct GPU architecture, as "native" is unsupported
- `-ngl|--n-gpu-layers` has to be set, in order to offload to layers to the GPU
    - `0` runs on CPU only
    - `33` is max for 7b model
    - `41` is max for 13b-chat model

## TODO
- [x] run 70b model
- [x] context size test
    - [x] use long document as context
    - [x] ask for details throughout the text (e.g. "what are the names of the characters?")
    - [x] test out different output format in the prompt, such as `json` or `csv`
- [ ] configuration matrix with test results
    - [ ] comments on the results, indicating success of output format, context size, etc.
    - [ ] eval time and GPU offloading for each
- [ ] run script that lets user select model, context size, output format, input document, prompt, etc. (probably easiest using environment variables)

## Long document comprehension and specific output format
- all non-chat models are basically useless, don't answer questions and proceed with writing another article
- chat models are better, able to deliver consistently formatted answers
- however, quality of answers is still very low
- 13b-chat works reasonably well
- 70b-chat-Q4_K_M just produces "end of text" immediately
- 70b-chat with higher quant is TBD

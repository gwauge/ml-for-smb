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
### Download & convert model to GGUF format
[Tutorial: How to convert HuggingFace model to GGUF format](https://github.com/ggerganov/llama.cpp/discussions/2948)

### Run model using llama.cpp
```bash
llama.cpp/main -m ../models/leolm-7b/leolm-7b-q8_0.gguf -n -2 -c 0 -ngl 40 -p "Das Wetter in Potsdam soll"
```

#### Parameters
- `-m` path to the model
- `-n` number of tokens to generate, `-2` means generate until context is full
- `-c` context size, `0` means  load from model (LeoLM uses `8192`)
- `-ngl` n GPU layers, offload [all `40` tensor layers](https://www.reddit.com/r/KoboldAI/comments/16op2jv/comment/k1m0i52/?utm_source=share&utm_medium=web2x&context=3) of gguf model to GPU
    - use `0` to run on CPU only
- `-p` prompt
- `--batch-size` can be used, however it does not seem to make a difference in inference time, LeoLM default is `512` (I believe)

## ONNX
### Convert model to ONNX
```bash
optimum-cli export onnx --model LeoLM/leo-hessianai-7b-chat models/leolm-7b-chat-onnx/
```

## Pitfalls
### llama.cpp
- has to be compiled with LLAMA_cuBLAS=1 to enable GPU support
    - might have to update the Makefile to correct GPU architecture, as "native" is unsupported
- `-ngl|--n-gpu-layers` has to be set, in order to offload to layers to the GPU (`40` is maximum)

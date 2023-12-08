## Download & convert model
[Tutorial: How to convert HuggingFace model to GGUF format](https://github.com/ggerganov/llama.cpp/discussions/2948)

## Run model using llama.cpp
```bash
llama.cpp/main -m ../models/leolm-7b/leolm-7b-q8_0.gguf -n -2 -c 0 -ngl 40 -p "Das Wetter in Potsdam soll"
```

### Parameters
- `-m` path to the model
- `-n` number of tokens to generate, `-2` means generate until context is full
- `-c` context size, `0` means  load from model (LeoLM uses `8192`)
- `-ngl` n GPU layers, offload [all `40` tensor layers](https://www.reddit.com/r/KoboldAI/comments/16op2jv/comment/k1m0i52/?utm_source=share&utm_medium=web2x&context=3) of gguf model to GPU
    - use `0` to run on CPU only
- `-p` prompt
- `--batch-size` can be used, however it does not seem to make a difference in inference time, LeoLM default is `512` (I believe)

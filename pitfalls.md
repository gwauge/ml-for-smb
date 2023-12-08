# Pitfalls

## llama.cpp
- has to be compiled with LLAMA_cuBLAS=1 to enable GPU support
    - might have to update the Makefile to correct GPU architecture, as "native" is unsupported
- `-ngl|--n-gpu-layers` has to be set, in order to offload to layers to the GPU (`40` is maximum)

# serve ollama and send process to the background
OLLAMA_MODELS=/workspaces/ml-for-smb/models/ ollama serve &
# run mixtral
ollama run mixtral:8x7b-instruct-v0.1-q3_K_M

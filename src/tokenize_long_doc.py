from transformers import AutoTokenizer

# Specify the path to the text file
file_path = "data/das_tapfere_schneiderlein.txt"

# Read the contents of the file
with open(file_path, "r") as file:
    text = file.read()

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("LeoLM/leo-hessianai-70b")

# Tokenize the text
tokens = tokenizer.tokenize(text)

# Print the tokens
print("num tokens:", len(tokens))

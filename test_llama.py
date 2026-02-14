from llama_cpp import Llama

llm = Llama(
    model_path="models/mistral.gguf",
    n_gpu_layers=-1
)

output = llm(
    "Q: What is Newton's First Law?\nA:",
    max_tokens=200
)

print(output["choices"][0]["text"])

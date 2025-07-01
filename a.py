import ollama
response = ollama.generate(
    model="mistral",
    prompt="What is Gautam Buddha University?"
)
print(response["response"])

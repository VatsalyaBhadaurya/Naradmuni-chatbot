import ollama
response = ollama.generate(
    model="llama3",
    prompt="What is Gautam Buddha University?"
)
print(response["response"])


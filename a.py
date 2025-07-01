import ollama
response = ollama.generate(
    model="llama3.2",
    prompt="What is Gautam Buddha University?"
)
print(response["response"])


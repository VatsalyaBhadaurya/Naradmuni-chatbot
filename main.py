import os
os.environ["CHROMA_TELEMETRY_ENABLED"] = "true"

import fitz  # PyMuPDF
import ollama
import chromadb
from tqdm import tqdm

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text, max_length=500):
    words = text.split()
    return [" ".join(words[i:i + max_length]) for i in range(0, len(words), max_length)]


def embed_documents(chunks):
    client = chromadb.PersistentClient(path="./embeddings")

    
    try:
        collection = client.get_collection("gbu_docs")
        print(" Collection already exists. Skipping embedding.")
        return
    except:
        print(" Creating new collection and embedding...")

    collection = client.create_collection(name="gbu_docs")

    for i, chunk in enumerate(tqdm(chunks, desc="Embedding Chunks")):
        try:
            response = ollama.embed(model="mxbai-embed-large", input=chunk)
            embeddings = response["embeddings"]
            collection.add(
                ids=[str(i)],
                embeddings=embeddings,
                documents=[chunk]
            )
        except Exception as e:
            print(f"❌ Failed to embed chunk {i}: {e}")

def answer_query(prompt):
    try:
        
        client = chromadb.PersistentClient(path="./embeddings")
        collection = client.get_collection("gbu_docs")

        
        print("🔍 Getting embedding...")
        query_embedding = ollama.embed(model="mxbai-embed-large", input=prompt)["embeddings"][0]

        
        print(" Querying vector DB...")
        results = collection.query(query_embeddings=[query_embedding], n_results=3)
        documents = results.get("documents", [])
        if not documents:
            return " No matching documents found."

        contexts = "\n".join([doc[0] for doc in documents])

        # Create the prompt
        final_prompt = f"""
You are a helpful university assistant for Gautam Buddha University. Use the context below to answer the question clearly and concisely.

Context:
{contexts}

Question:
{prompt}

Answer:
"""

        
        print(" Generating answer...")
        output = ollama.generate(model="mistral", prompt=final_prompt)

        return output.get("response", "⚠️ No response generated.")

    except Exception as e:
        return f"❌ Error: {e}"



def run_pipeline():
    pdf_folder = "./data"
    all_chunks = []

    print("\n Reading and chunking PDFs...")
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)
            raw_text = extract_text_from_pdf(pdf_path)
            chunks = chunk_text(raw_text)
            all_chunks.extend(chunks)

    print(f"\n📄 Total chunks to embed: {len(all_chunks)}")

    if not all_chunks:
        print(" No chunks found to embed. Please check your PDF files.")
        return

    embed_documents(all_chunks)
    print("\n Document embedding complete. You can now query the system.\n")


if __name__ == "__main__":
    run_pipeline()
    while True:
        query = input("❓ Ask a question (or type 'exit'): ")
        if query.lower() == "exit":
            break
        answer = answer_query(query)
        print("\n Answer:\n", answer, "\n")

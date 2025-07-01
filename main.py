import os
import sys
import fitz
import ollama
import chromadb
from tqdm import tqdm

# Redirect stderr to suppress telemetry messages
class DummyFile:
    def write(self, x): pass
    def flush(self): pass

sys.stderr = DummyFile()

def extract_text_from_pdf(pdf_path):
    with fitz.open(pdf_path) as doc:
        return " ".join(page.get_text() for page in doc)

def extract_text_from_txt(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as file:
        return file.read()

def chunk_text(text, max_length=500):
    words = text.split()
    return [" ".join(words[i:i + max_length]) for i in range(0, len(words), max_length)]

def embed_documents(chunks):
    client = chromadb.PersistentClient(path="./embeddings")

    try:
        client.delete_collection("gbu_docs")
    except:
        pass

    collection = client.create_collection(name="gbu_docs")
    successful_embeds = 0

    for i, chunk in enumerate(tqdm(chunks, desc="Embedding Chunks")):
        try:
            embeddings = ollama.embed(model="mxbai-embed-large", input=chunk)["embeddings"]
            collection.add(
                ids=[str(i)],
                embeddings=embeddings,
                documents=[chunk]
            )
            successful_embeds += 1
        except Exception as e:
            print(f"❌ Failed to embed chunk {i}: {e}")
    
    print(f"\n✅ Successfully embedded {successful_embeds} out of {len(chunks)} chunks")
    return successful_embeds > 0

def answer_query(prompt):
    try:
        client = chromadb.PersistentClient(path="./embeddings")
        collection = client.get_collection("gbu_docs")

        query_embedding = ollama.embed(model="mxbai-embed-large", input=prompt)["embeddings"]
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=3
        )
        
        documents = results["documents"]
        if not documents or not documents[0]:
            return "No matching documents found."

        context = "\n".join(doc[0] for doc in documents)
        prompt = f"""You are a helpful university assistant for Gautam Buddha University. Use the context below to answer the question clearly and concisely.

Context:
{context}

Question:
{prompt}

Answer:"""

        return ollama.generate(model="mistral", prompt=prompt).get("response", "No response generated.")

    except Exception as e:
        return f"Error: {e}"

def main():
    data_folder = "./data"
    chunks = []

    print("\nReading and processing documents...")
    for filename in os.listdir(data_folder):
        file_path = os.path.join(data_folder, filename)
        try:
            if filename.endswith(".pdf"):
                print(f"Processing PDF: {filename}")
                chunks.extend(chunk_text(extract_text_from_pdf(file_path)))
            elif filename.endswith(".txt"):
                print(f"Processing TXT: {filename}")
                chunks.extend(chunk_text(extract_text_from_txt(file_path)))
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

    if not chunks:
        print("No text found in documents. Please check your files.")
        return

    print(f"\n📄 Total chunks to embed: {len(chunks)}")
    
    if not embed_documents(chunks):
        print("\n❌ Failed to embed documents. Please check the errors above.")
        return

    print("\nDocument embedding complete. You can now query the system.\n")

    while True:
        query = input("❓ Ask a question (or type 'exit'): ")
        if query.lower() == "exit":
            break
        print("\nAnswer:", answer_query(query), "\n")

if __name__ == "__main__":
    main()

import os
import sys
import fitz
import ollama
import chromadb
from tqdm import tqdm

# Dummy file to suppress stderr output, jaise main apne dosto ki bakwas sunta hoon
class DummyFile:
    def write(self, x): pass  
    def flush(self): pass     # Flush karo ya nahi, humko kya fark padta hai

sys.stderr = DummyFile()

def extract_text_from_pdf(pdf_path):  #pdf mat use karo yrr pls haath jor raha hu :(
    with fitz.open(pdf_path) as doc:
        return " ".join(page.get_text() for page in doc)

def extract_text_from_txt(txt_path):     # hum txt file se text nikaal rahe hai
    with open(txt_path, 'r', encoding='utf-8') as file:
        return file.read()

def chunk_text(text, max_length=500):    # text ko tod rahe hain jaise main...never mind
    words = text.split()
    return [" ".join(words[i:i + max_length]) for i in range(0, len(words), max_length)]

def embed_documents(chunks):
    client = chromadb.PersistentClient(path="./embeddings")

    try:    # deleting old data, jaise main usseke purane messages delete karta hoon
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
            print(f"❌chunk failed {e}")
    
    print(f"\n✅ Waah! {successful_embeds} chunks embed ho gaye, total {len(chunks)} mein se")
    return successful_embeds > 0

def answer_query(prompt):
    try:                                                # ollama se query ka jawab le rahe hain
        client = chromadb.PersistentClient(path="./embeddings")
        collection = client.get_collection("gbu_docs")

        query_embedding = ollama.embed(model="mxbai-embed-large", input=prompt)["embeddings"]
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=3
        )
        
        documents = results["documents"]
        if not documents or not documents[0]:
            return "No matching docs"

        context = "\n".join(doc[0] for doc in documents)
        prompt = f"""You are a helpful university assistant for Gautam Buddha University. Use the context below to answer the question clearly and concisely.

Context:
{context}

Question:
{prompt}

Answer:"""

        return ollama.generate(model="mistral", prompt=prompt).get("response", "No Response")

    except Exception as e:
        return f"Error ho gaya bhai: {e} huihuihi"

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

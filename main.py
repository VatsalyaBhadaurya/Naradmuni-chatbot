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

def get_embedding(text):
    """Get embeddings using Ollama's embeddings endpoint with nomic-embed-text model"""
    try:
        # First, make sure we have the model
        ollama.pull('nomic-embed-text')
        
        # Get embeddings
        response = ollama.embeddings(
            model='nomic-embed-text',
            prompt=text
        )
        return response.get('embedding', None)
    except Exception as e:
        print(f"Error getting embedding: {str(e)}")
        return None

def embed_documents(chunks):
    try:
        client = chromadb.PersistentClient(path="./embeddings")

        # Delete existing collection if it exists
        try:
            client.delete_collection("gbu_docs")
        except:
            pass

        # Create new collection with metadata
        collection = client.create_collection(
            name="gbu_docs",
            metadata={"hnsw:space": "cosine"}  # Specify distance metric
        )

        successful_embeds = 0
        for i, chunk in enumerate(tqdm(chunks, desc="Embedding Chunks")):
            try:
                embeddings = get_embedding(chunk)
                if embeddings is None:
                    print(f"❌ Chunk {i} failed: No embeddings returned")
                    continue
                    
                collection.add(
                    ids=[str(i)],
                    embeddings=embeddings,
                    documents=[chunk],
                    metadatas=[{"source": "gbu_docs", "chunk_id": str(i)}]
                )
                successful_embeds += 1
            except Exception as e:
                print(f"❌ Chunk {i} failed: {str(e)}")
        
        print(f"\n✅ Waah! {successful_embeds} chunks embed ho gaye, total {len(chunks)} mein se")
        return successful_embeds > 0
    except Exception as e:
        print(f"❌ Error in embed_documents: {str(e)}")
        return False
    
    
# Function to check if the query is relevant to GBU context or just random bakwas
def is_relevant_query(prompt, threshold=0.6):
    try:
        client = chromadb.PersistentClient(path="./embeddings")
        collection = client.get_collection("gbu_docs")
        query_embedding = get_embedding(prompt)
        if query_embedding is None:
            return False

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=1,
            include=["distances"]
        )

        # Distance close to 0 = very similar. 1 = not similar (for cosine)
        distance = results.get("distances", [[1]])[0][0]
        return distance < threshold
    except Exception as e:
        print(f"⚠️ Error in is_relevant_query: {str(e)}")
        return False


def answer_query(prompt):
    try:
        client = chromadb.PersistentClient(path="./embeddings")
        collection = client.get_collection("gbu_docs")
        
        if not is_relevant_query(prompt):
            return "I don't know about that, ask me about GBU"

        query_embedding = get_embedding(prompt)
        if query_embedding is None:
            return "Sorry, I couldn't process your question at the moment"

        results = collection.query(
            query_embeddings=[query_embedding],  # Wrap in list as required by ChromaDB
            n_results=3
        )
        
        documents = results["documents"]
        if not documents or not documents[0]:
            return "No matching docs found for your query"

        context = "\n".join(doc[0] for doc in documents)
        prompt = f"""You are a helpful university assistant for Gautam Buddha University. Use the context below to answer the question clearly and concisely.

Context:
{context}

Question:
{prompt}

Answer:"""

        response = ollama.generate(model="mistral", prompt=prompt)
        if not response or "response" not in response:
            return "Sorry, I couldn't generate a response at the moment"
            
        return response["response"]

    except Exception as e:
        error_msg = str(e)
        print(f"Error in answer_query: {error_msg}")  # Log the error
        if "no such column" in error_msg:
            return "Database error occurred. Please restart the server to rebuild the database."
        return f"Error ho gaya bhai: {error_msg} huihuihi"

def main():
    data_folder = "./data"
    chunks = []

    # Check if data folder exists
    if not os.path.exists(data_folder):
        print(f"❌ Data folder not found: {data_folder}")
        return

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
            print(f"❌ Error processing {filename}: {str(e)}")

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

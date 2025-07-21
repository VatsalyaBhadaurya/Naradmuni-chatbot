import os
import sys
import fitz
import ollama
import chromadb
from tqdm import tqdm
import whisper
import tempfile
from werkzeug.utils import secure_filename

from rapidfuzz import process

# Important GBU-specific terms you care about
important_keywords = [
    "Gautam Buddha University",
    "GBU(Gautam Buddha University)",
    "B.Tech CSE",
    "B.Tech AI",
    "M.Tech",
    "MBA",
    "PhD",
    "hostel",
    "placement",
    "campus",
    "fees",
    "scholarship",
    "exam",
    "UG(Undergraduate)",
    "PG(Postgraduate)",
    "faculty",
    "NAAC",
    "NIRF",
    "prospectus",
    "department",
    "library",
    "canteen",
    "engineering",
    "contact",
]

# Whisper karne ke liye whisper model load kar rahe hain, eaves drop nahi karega pakka promise
whisper_model = whisper.load_model("small")  # or "base", "medium", "large"

def transcribe_audio(file):
    try:
        filename = secure_filename(file.filename)
        ext = os.path.splitext(file.filename)[1] or ".webm"
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_audio:
            file.save(temp_audio.name)
            result = whisper_model.transcribe(temp_audio.name)
            return result["text"]
    except Exception as e:
        return f"Error during transcription: {str(e)}"

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
# Added lightweight autocorrect and fuzzy matching for better query handling(optional par contextual 🎓📝 prompt ke liye madad karega)
from fuzzywuzzy import fuzz
from textblob import TextBlob

def correct_prompt(user_prompt, threshold=85):
    """
    Fuzzy correct only known important keywords to prevent distortion of short or generic prompts.
    """
    if len(user_prompt.strip().split()) <= 2 and user_prompt.lower() in ["hi", "hello", "hey", "how are you"]:
        return user_prompt  # don't alter common greetings

    corrected = user_prompt
    words = user_prompt.split()
    for word in words:
        result = process.extractOne(word, important_keywords, score_cutoff=threshold)
        if result:
            match, score, _ = result
            if match.lower() != word.lower():
                corrected = corrected.replace(word, match)

    return corrected



def is_relevant_query(prompt, threshold=0.35): 
    try:
        client = chromadb.PersistentClient(path="./embeddings")
        collection = client.get_collection("gbu_docs")

        corrected_prompt = correct_prompt(prompt)
        print(f"✅ Corrected Prompt (for relevance): {corrected_prompt}")

        query_embedding = get_embedding(corrected_prompt)
        if query_embedding is None:
            return False

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3,
            include=["documents", "distances"]
        )

        documents = results.get("documents", [[]])[0]
        distances = results.get("distances", [[1]])[0]

        if not documents:
            return False

        # Check fuzzy similarity against top doc
        top_doc = documents[0]
        similarity = fuzz.token_set_ratio(corrected_prompt, top_doc)

        print(f"🔍 Embedding similarity: {1 - distances[0]}, Fuzzy similarity: {similarity}")

        return (1 - distances[0]) > threshold or similarity > 65
    except Exception as e:
        print(f"⚠️ Error in is_relevant_query: {str(e)}")
        return False


def answer_query(prompt):
    try:
        # ⛔ Direct responses for common greetings
        cleaned = prompt.lower().strip()
        if any(cleaned.startswith(greet) for greet in ["hi", "hello", "hey", "how are you", "good morning", "good evening"]):
            return "Hello! How can I help you today?"

        client = chromadb.PersistentClient(path="./embeddings")
        collection = client.get_collection("gbu_docs")
        
        if not is_relevant_query(prompt):  # use original prompt
            return "I don't know about that, ask me about GBU"

        query_embedding = get_embedding(prompt)  # again, original prompt
        if query_embedding is None:
            return "Sorry, I couldn't process your question at the moment"

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )
        
        documents = results.get("documents", [[]])[0]
        if not documents:
            return "No matching docs found for your query"

        context = "\n".join(doc for doc in documents)

        final_prompt = f"""You are a helpful university assistant for Gautam Buddha University. Use the context below to answer the question clearly and concisely.

Context:
{context}

Question:
{prompt}

Answer:"""

        response = ollama.generate(model="mistral", prompt=final_prompt)
        if not response or "response" not in response:
            return "Sorry, I couldn't generate a response at the moment"
            
        return response["response"]

    except Exception as e:
        error_msg = str(e)
        print(f"Error in answer_query: {error_msg}")
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

import os
import sys
import re     # Regex ke liye - text ko samajhdaari se todne ke liye
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

def chunk_text_semantic(text, max_sentences=5, overlap_sentences=1):  # ab semantic chunking karenge, samajhdaar ban gaye hain
    """
    Semantic chunking - ab text ko samajhdaari se todenge, jaise main apne dost ki baatein sunta hoon
    """
    import re
    
    # Pehle sentences mein break karo, jaise family WhatsApp group ke messages
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]  # Saaf kar do, gandagi nahi chahiye
    
    chunks = []
    
    # Har section ko identify karo - numbered headers dekho jaise 1., 2., 3.
    current_section = []
    section_chunks = []
    
    for sentence in sentences:
        # Agar numbered header hai toh naya section shuru kar do
        if re.match(r'^\d+\.\s+[A-Z]', sentence):  # Jaise "1. General Overview" 
            if current_section:  # Purana section complete kar do pehle
                section_chunks.append(" ".join(current_section))
            current_section = [sentence]  # Naya section shuru karo
        else:
            current_section.append(sentence)  # Same section mein add kar do
    
    # Last section ko bhi add karna bhool mat jao, jaise main kabhi nahi bhulta
    if current_section:
        section_chunks.append(" ".join(current_section))
    
    # Ab har section ko further break karo agar bahut bada hai
    final_chunks = []
    for section in section_chunks:
        words = section.split()
        if len(words) > 400:  # Agar section bahut lamba hai toh tod do
            # Sub-sections by bullet points ya colons se tod do
            sub_parts = re.split(r'\n(?=[-*•]|\w+:)', section)
            for part in sub_parts:
                if part.strip():  # Empty nahi hona chahiye
                    final_chunks.append(part.strip())
        else:
            final_chunks.append(section)  # Chhota hai toh aise hi rakh do
    
    # Overlap add karo taaki context miss na ho jaye, jaise mere paas memories overlap karti hain
    overlapped_chunks = []
    for i, chunk in enumerate(final_chunks):
        current_chunk = chunk
        
        # Previous chunk ka thoda sa context add kar do agar hai toh
        if i > 0 and overlap_sentences > 0:
            prev_sentences = re.split(r'(?<=[.!?])\s+', final_chunks[i-1])
            overlap_text = " ".join(prev_sentences[-overlap_sentences:])
            current_chunk = overlap_text + " " + current_chunk
        
        overlapped_chunks.append(current_chunk)
    
    print(f"📝 Semantic chunking complete! {len(overlapped_chunks)} chunks banaye, ab chatbot samajhdaar ban jayega")
    return overlapped_chunks

def chunk_text(text, max_length=500):    # Purana wala method - ab use nahi karenge, semantic chunking ka zamana hai
    # Backup ke liye rakh rahe hain, agar semantic chunking fail ho jaye toh
    words = text.split()
    return [" ".join(words[i:i + max_length]) for i in range(0, len(words), max_length)]

def embed_documents(chunks):
    client = chromadb.PersistentClient(path="./embeddings")  # Embeddings ka ghar banaya

    try:    # Purana data delete kar rahe hain, jaise main usseke purane messages delete karta hoon
        client.delete_collection("gbu_docs")
        print("🗑️ Purana collection delete kar diya, ab naya banayenge")
    except:
        print("📁 Koi purana collection nahi mila, fresh start kar rahe hain")
        pass

    collection = client.create_collection(name="gbu_docs")  # Naya collection banaya, bilkul fresh
    successful_embeds = 0
    
    for i, chunk in enumerate(tqdm(chunks, desc="Semantic Chunks ko Embed kar rahe hain")):
        try:
            # Har chunk ko vector mein convert kar rahe hain, jaise thoughts ko words mein
            embeddings = ollama.embed(model="mxbai-embed-large", input=chunk)["embeddings"]
            collection.add(
                ids=[str(i)],           # Har chunk ka unique ID, jaise Aadhar card
                embeddings=embeddings,   # Vector representation - AI ki language
                documents=[chunk]        # Original text bhi save kar rahe hain
            )
            successful_embeds += 1
        except Exception as e:
            print(f"❌ Yeh chunk embed nahi hua bhai: {e}")  # Koi chunk problematic tha
    
    print(f"\n✅ Zabardast! {successful_embeds} semantic chunks successfully embed ho gaye, total {len(chunks)} mein se")
    print(f"🧠 Ab chatbot ko {successful_embeds} different topics ke baare mein pata hai")
    return successful_embeds > 0

def answer_query(user_query):
    try:                                                # Ab semantic chunks use karke better answer denge
        client = chromadb.PersistentClient(path="./embeddings")  # Embeddings database ko access kar rahe hain
        collection = client.get_collection("gbu_docs")          # Apna GBU docs collection

        # User ke query ko bhi vector mein convert kar rahe hain, jaise uske thoughts ko samajh rahe hain
        query_embedding = ollama.embed(model="mxbai-embed-large", input=user_query)["embeddings"]
        
        # Sabse similar chunks dhundh rahe hain, jaise similar minded friends dhundte hain
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=3  # Top 3 most relevant semantic chunks lenge
        )
        
        documents = results["documents"]
        if not documents or not documents[0]:
            return "🤔 Koi matching information nahi mila boss, kuch aur poocho"

        # Sabse relevant semantic chunks ko combine kar rahe hain
        context = "\n\n".join(doc for doc in documents[0])  # Har chunk ko alag line mein
        
        # Ab LLM ko proper context de rahe hain semantic chunks ke saath
        enhanced_prompt = f"""You are a knowledgeable assistant for Gautam Buddha University (GBU). Use the provided context to answer questions accurately and helpfully.

CONTEXT (from semantic chunks):
{context}

STUDENT QUESTION: {user_query}

INSTRUCTIONS:
- Answer based ONLY on the provided context
- Be specific and detailed
- Use bullet points for lists
- If information is not in context, say so clearly
- Be helpful and friendly

ANSWER:"""

        # Mistral se final answer generate kar rahe hain
        response = ollama.generate(model="mistral", prompt=enhanced_prompt).get("response", "Kuch problem hai, dobara try karo")
        
        print(f"🎯 Query: '{user_query}' ka jawab semantic chunks se mil gaya!")
        return response

    except Exception as e:
        return f"❌ Oops! Error ho gaya bhai: {e} - Thoda wait kar ke dobara try karo"

def main():
    data_folder = "./data"
    chunks = []

    print("\nReading and processing documents...")
    for filename in os.listdir(data_folder):
        file_path = os.path.join(data_folder, filename)
        try:
            if filename.endswith(".pdf"):
                print(f"📄 Processing PDF: {filename} - PDF se text nikaal ke semantic chunks banayenge")
                text = extract_text_from_pdf(file_path)
                chunks.extend(chunk_text_semantic(text))  # Ab semantic chunking use kar rahe hain
            elif filename.endswith(".txt"):
                print(f"📝 Processing TXT: {filename} - Text file ko samajhdaari se tod rahe hain")
                text = extract_text_from_txt(file_path)
                chunks.extend(chunk_text_semantic(text))  # Semantic chunking, simple chunking nahi
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

    if not chunks:
        print("No text found in documents. Please check your files.")
        return

    print(f"\n📊 Total semantic chunks ready for embedding: {len(chunks)}")
    print("🚀 Ab in chunks ko vectors mein convert karenge - AI magic shuru!")
    
    if not embed_documents(chunks):
        print("\n❌ Embedding fail ho gaya yaar! Upar ke errors dekho kya problem hai.")
        return

    print("\n🎉 Semantic chunking aur embedding complete! Ab chatbot bilkul ready hai.")
    print("💬 GBU ke baare mein jo bhi puchna hai, poocho - ab intelligent answers milenge!\n")

    while True:
        query = input("🎓 GBU ke baare mein kya jaanna hai? (ya 'exit' type karo): ")
        if query.lower() == "exit":
            print("👋 Bye bye! Phir aana GBU ke baare mein puchne!")
            break
        print(f"\n🤖 AI Assistant: {answer_query(query)}\n")
        print("-" * 80)  # Separator line for better readability

if __name__ == "__main__":
    main()

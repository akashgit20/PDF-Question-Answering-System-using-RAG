import os
import fitz  # PyMuPDF for extracting text from PDFs
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Set Gemini API Key
genai.configure(api_key="YOUR_GEMINI_API_KEY_HERE")

def extract_text_from_pdf(pdf_path):
    """Extract text from a given PDF file."""
    doc = fitz.open(pdf_path)
    return "\n".join([page.get_text("text") for page in doc])

def split_text_into_chunks(text, chunk_size=1000, chunk_overlap=100):
    """Split text into chunks for better processing in embeddings."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_text(text)

def create_vector_store(text_chunks):
    """Convert text chunks into vector embeddings and store in FAISS."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(text_chunks, embeddings)

def query_gemini(prompt):
    """Query Gemini API to generate answers."""
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt)
    return response.text if response else "No response from Gemini."

def main():
    pdf_folder = "pdfs"  # Folder containing PDFs
    
    # Extract and process PDF text
    all_text = "\n".join([extract_text_from_pdf(os.path.join(pdf_folder, pdf_file))
                            for pdf_file in os.listdir(pdf_folder) if pdf_file.endswith(".pdf")])
    
    print("Processing PDFs...")
    text_chunks = split_text_into_chunks(all_text)
    vector_store = create_vector_store(text_chunks)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    
    print("Ready to answer questions! Type 'Quit' to exit.")
    while True:
        query = input("Ask a question: ")
        if query.lower() == "quit":
            print("Goodbye!")
            break
        
        # Retrieve relevant chunks
        relevant_chunks = retriever.invoke(query)
        context = "\n".join([doc.page_content for doc in relevant_chunks])
        
        if context.strip():  # If relevant chunks exist
            prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        else:
            prompt = f"Question: {query}\nAnswer:"
        
        print("Generating response...")
        answer = query_gemini(prompt) 
        
        print("\nAnswer:", answer, "\n")

if __name__ == "__main__":
    main()

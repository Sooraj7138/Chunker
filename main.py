import json
import os
import re
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from itertools import chain
from unstructured.partition.auto import partition
from transformers import AutoTokenizer
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

# Initialize ChromaDB Client
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Use Sentence Transformer Embedding Function
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-large-en-v1.5"
)

collection = chroma_client.get_or_create_collection(
    name="pdf_chunks", 
    embedding_function=sentence_transformer_ef
)

def clean_text(text):
    text = text.replace("\x00", "")    # remove null chars
    # text = re.sub(r"\s+", " ", text)   # DONT normalize whitespace here, we need it for splitting
    return text.strip()

def add_newlines(text):
    text = re.sub(r"(?<!\d)\.\s+", ".\n", text)
    text = re.sub(r"\?\s+", "?\n", text)
    text = re.sub(r"!\s+", "!\n", text)
    text = re.sub(
        r"(?<=\s)(\d+)\.\s+(?=[A-Z])",
        r"\n\1. ",
        text
    )
    text = text.replace("• ", "\n• ")
    return text

def chunk_text_by_words(text, words_per_chunk=800, overlap=160):
    # Split by any whitespace but keep the words
    words = text.split()
    chunks = []
    
    if not words:
        return []

    # Calculate the step size
    step = words_per_chunk - overlap
    
    # Ensure step is at least 1 to avoid infinite loop
    if step <= 0:
        step = 1

    for i in range(0, len(words), step):
        chunk_words = words[i:i + words_per_chunk]
        chunk = " ".join(chunk_words)
        chunks.append(chunk)
        
        # If we have reached or passed the end of the text, stop
        if i + words_per_chunk >= len(words):
            break
            
    return chunks

def process_single_pdf(file_path, output_dir):
    try:
        # Use hi_res strategy to identify tables
        elements = partition(filename=file_path, languages=["eng"], strategy='hi_res')
        
        # Build content by including table HTML when available to preserve structure in chunks
        content_parts = []
        for el in elements:
            if el.category == "Table":
                # Ensure table_html is a string even if text_as_html is None
                table_html = getattr(el.metadata, "text_as_html", None) or str(el)
                content_parts.append(table_html)
            else:
                content_parts.append(str(el))
        
        content = "\n\n".join(content_parts)
        
        if not content.strip():
            print(f"⚠️ No content extracted from {file_path}")
            return

        cleaned_text = clean_text(content)
        # Use word-based chunking with overlap
        chunks = chunk_text_by_words(cleaned_text, words_per_chunk=800, overlap=160)

        chunk_id = 0
        for chunk in chunks:
            formatted_chunk = add_newlines(chunk)
            wrapped_chunk = f"<doc_start>\n{formatted_chunk}\n<doc_end>\n"
            
            # Save to local file
            output_filename = f"chunk_{chunk_id:06d}.txt"
            with open(os.path.join(output_dir, output_filename), "w", encoding="utf-8") as f:
                f.write(wrapped_chunk)
            
            # Save to ChromaDB
            collection.upsert(
                documents=[formatted_chunk],
                metadatas=[{
                    "source": os.path.basename(file_path),
                    "chunk_id": chunk_id,
                }],
                ids=[f"{os.path.basename(file_path)}_{chunk_id}"]
            )
            
            chunk_id += 1

        print(f"✅ Processed {os.path.basename(file_path)}. \nTotal chunks saved: {chunk_id}")
        print("----------------------------------------------------------------------------------------------")
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")

def retrieve_chunks(query, n_results=5):
    """
    Retrieve chunks from ChromaDB based on query and optional chunking_type filter.
    """
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
    )
    return results

if __name__ == "__main__":
    vts_folder = "./VTS"
    if not os.path.exists(vts_folder):
        print(f"❌ Folder {vts_folder} not found.")
    else:
        # for file in os.listdir(vts_folder):
        #     if file.lower().endswith(".pdf"):
        #         file_path = os.path.join(vts_folder, file)
        #         base_dir_name = file.replace(".pdf", "")
        #         output_path = os.path.join("Output", base_dir_name)
        #         os.makedirs(output_path, exist_ok=True)
                
        #         print(f"Processing: {file}...")
        #         process_single_pdf(file_path, output_path)
        
        # Example Q&A Retrieval
        print("\n--- Question & Answer Retrieval Test ---")
        question = "Radio communications message structure"
        print(f"Question: {question}")
        
        search_results = retrieve_chunks(question, n_results=5)
        
        if search_results:
            for i in range(len(search_results['ids'][0])):
                doc_id = search_results['ids'][0][i]
                content = search_results['documents'][0][i]
                distance = search_results['distances'][0][i] if 'distances' in search_results else "N/A"
                
                print(f"\n--- Result {i+1} (ID: {doc_id}, Distance: {distance:.4f}) ---")
                print(f"Content Preview: {content[:1000]}...")
                print("-" * 50)
        else:
            print("No results found.")

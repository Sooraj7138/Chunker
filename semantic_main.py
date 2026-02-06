import json
import os
import re
import numpy as np
import chromadb
from chromadb.utils import embedding_functions
from unstructured.partition.auto import partition
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# Initialize ChromaDB Client
chroma_client = chromadb.PersistentClient(path="./chroma_db_semantic")

# Initialize LangChain Embeddings (HuggingFace Local)
# Using BGE-Large for high-accuracy retrieval
embeddings_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={'device': 'cpu'}
)

# Initialize LangChain LLM (Groq - Llama 3)
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# Initialize Cross-Encoder for reranking
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Wrapper for ChromaDB to use LangChain embeddings
class LangChainEmbeddingFunction(chromadb.EmbeddingFunction):
    def __call__(self, input):
        if isinstance(input, str):
            input = [input]
        return embeddings_model.embed_documents(input)

collection = chroma_client.get_or_create_collection(
    name="semantic_pdf_chunks_llama", # New name for new embedding model
    embedding_function=LangChainEmbeddingFunction()
)

def clean_text(text):
    text = text.replace("\x00", "")
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

def semantic_chunking(elements, breakpoint_percentile_threshold=95):
    """
    Groups elements into chunks based on semantic similarity between consecutive elements.
    """
    if not elements:
        return []
    if len(elements) < 2:
        return elements

    # 1. Embed all elements using LangChain model
    embeddings = embeddings_model.embed_documents(elements)

    # 2. Calculate cosine distances between consecutive elements
    distances = []
    for i in range(len(embeddings) - 1):
        similarity = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
        distance = 1 - similarity
        distances.append(distance)

    # 3. Identify breakpoints based on distance threshold
    threshold = np.percentile(distances, breakpoint_percentile_threshold)
    
    chunks = []
    current_chunk = [elements[0]]
    
    for i, distance in enumerate(distances):
        if distance > threshold:
            # End current chunk and start a new one
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [elements[i+1]]
        else:
            current_chunk.append(elements[i+1])
            
    # Add the last chunk
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
        
    return chunks

def process_single_pdf(file_path, output_dir):
    try:
        # Use hi_res strategy to identify tables
        elements = partition(filename=file_path, languages=["eng"], strategy='hi_res')
        
        # Build content elements by binding titles to following content for better context
        content_parts = []
        current_context = ""
        for el in elements:
            if el.category in ["Title", "Header", "Headline"]:
                current_context = str(el)
                # Don't add titles as standalone elements if they'll be bound to next el
                continue
            
            part = ""
            if el.category == "Table":
                table_content = getattr(el.metadata, "text_as_html", None) or str(el)
                # Prepend context (title) to the table
                part = f"{current_context}\n{table_content}" if current_context else table_content
            else:
                text_content = str(el)
                part = f"{current_context}\n{text_content}" if current_context else text_content
            
            cleaned_part = clean_text(part)
            if cleaned_part:
                content_parts.append(cleaned_part)
                # Keep current_context for multiple elements until a new title appears
        
        if not content_parts:
            print(f"⚠️ No content extracted from {file_path}")
            return
        
        # Perform Semantic Chunking on elements
        print(f"Applying element-based semantic chunking to {os.path.basename(file_path)}...")
        chunks = semantic_chunking(content_parts, breakpoint_percentile_threshold=90)

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
                    "method": "semantic"
                }],
                ids=[f"semantic_{os.path.basename(file_path)}_{chunk_id}"]
            )
            
            chunk_id += 1

        print(f"✅ Processed {os.path.basename(file_path)}. \nTotal semantic chunks saved: {chunk_id}")
        print("----------------------------------------------------------------------------------------------")
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")

def retrieve_chunks(query, n_results=5):
    """
    Retrieve chunks from ChromaDB.
    """
    print(f"DEBUG: Collection count: {collection.count()}")
    print(f"DEBUG: Querying with keywords: {query}")
    
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
    )
    
    # Debug results
    if results and results['documents'][0]:
        print(f"DEBUG: Found {len(results['documents'][0])} results.")
    else:
        print("DEBUG: No results returned from ChromaDB.")
        
    return results

def refine_query(user_question):
    """
    Use LLM to analyze the scenario and extract precise technical keywords for vector search.
    """
    print(f"Refining query for: {user_question}")
    
    prompt = ChatPromptTemplate.from_template(
        "As a VTS and IALA Guideline expert, analyze the following maritime scenario question: '{question}'\n\n"
        "Your task is to identify and output ONLY the most relevant technical keywords and IALA guideline "
        "references to retrieve the correct technical documentation from a vector database.\n\n"
        "Focus on extracting:\n"
        "1. Technical groups or alert categories (e.g., 'Incident response group', 'Real-time alert')\n"
        "2. Specific maritime events or vessel statuses (e.g., 'propulsion loss', 'AIS status change')\n"
        "3. Any table names or functional requirements implied.\n\n"
        "Output ONLY the keywords as a single line, separated by spaces."
    )
    
    chain = prompt | llm | StrOutputParser()
    refined = chain.invoke({"question": user_question})
    
    print(f"Refined Query: {refined.strip()}")
    return refined.strip()

def generate_scenario_response(scenario_question, context_chunks):
    """
    Generate a detailed scenario-based response using retrieved chunks as context.
    """
    context_text = "\n\n".join(context_chunks)
    
    prompt = ChatPromptTemplate.from_template(
        "You are a professional VTS (Vessel Traffic Services) expert. Answer the following scenario-based "
        "question using ONLY the provided technical context. If the context does not provide enough information, "
        "state what specific information is missing from the IALA guidelines.\n\n"
        "CONTEXT:\n{context}\n\n"
        "SCENARIO QUESTION:\n{question}\n\n"
        "ANSWER:"
    )
    
    print("Generating scenario-based response...")
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({
        "context": context_text,
        "question": scenario_question
    })
    
    return response.strip()

if __name__ == "__main__":
    vts_folder = "./VTS"
    if not os.path.exists(vts_folder):
        print(f"❌ Folder {vts_folder} not found.")
    else:
        # for file in os.listdir(vts_folder):
        #     if file.lower().endswith(".pdf"):
        #         file_path = os.path.join(vts_folder, file)
        #         base_dir_name = file.replace(".pdf", "")
        #         output_path = os.path.join("Semantic_Output", base_dir_name)
        #         os.makedirs(output_path, exist_ok=True)
                
        #         print(f"Processing: {file}...")
        #         process_single_pdf(file_path, output_path)

        # Example Scenario-based Q&A
        print("\n--- AI-Powered Scenario-Based Retrieval ---")
        scenario_question = "During a period of heavy weather, a bulk carrier at the anchorage begins to move at 1.5 knots, suggesting its anchor is dragging. What is the specific Real-time alert and its associated prerequisite for this situation?"
        
        # 1. Refine query for better retrieval
        search_query = refine_query(scenario_question)
        
        # 2. Retrieve relevant chunks
        search_results = retrieve_chunks(search_query, n_results=5)
        
        if search_results and search_results['documents'][0]:
            retrieved_docs = search_results['documents'][0]
            
            # 3. Generate final scenario-based response
            final_answer = generate_scenario_response(scenario_question, retrieved_docs)
            
            print("\n" + "="*50)
            print("FINAL SCENARIO RESPONSE:")
            print("="*50)
            print(final_answer)
            print("="*50)
        else:
            print("No relevant context found to generate a response.")

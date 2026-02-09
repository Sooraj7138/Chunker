from neo4j import GraphDatabase
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
from pydantic import BaseModel
import ollama
from collections import defaultdict
from neo4j_graphrag.retrievers import QdrantNeo4jRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from unstructured.partition.auto import partition
from typing import List, Dict, Any, Optional
from docx import Document
import uuid
import os
import json
import re

# Load environment variables
load_dotenv()

# Get credentials from environment variables
qdrant_key = os.getenv("QDRANT_KEY")
qdrant_url = os.getenv("QDRANT_URL")
neo4j_uri = os.getenv("NEO4J_URI")
neo4j_username = os.getenv("NEO4J_USERNAME")
neo4j_password = os.getenv("NEO4J_PASSWORD")

# Initialize Neo4j driver
neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password))

# Initialize Qdrant client
qdrant_client = QdrantClient(
    url=qdrant_url,
    api_key=qdrant_key
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,        # safe for nomic-embed-text
    chunk_overlap=50
)

def clean_text(text):
    text = text.replace("\x00", "")    # remove null chars
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


class single(BaseModel):
    node: str
    target_node: Optional[str] = None
    relationship: Optional[str] = None

class GraphComponents(BaseModel):
    graph: List[single]

def ollama_llm_parser(prompt):
    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "system",
                "content": 
                """ You are a precise graph relationship extractor. Extract all 
                    relationships from the text and format them as a JSON object 
                    with this exact structure:
                    {
                        "graph": [
                            {"node": "Person/Entity", 
                             "target_node": "Related Entity", 
                             "relationship": "Type of Relationship"},
                            ...more relationships...
                        ]
                    }
                    Include ALL relationships mentioned in the text, including 
                    implicit ones. Be thorough and precise. Return ONLY the JSON object. """
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        format="json"
    )
    
    return GraphComponents.model_validate_json(response['message']['content'])

def extract_graph_components(raw_data):
    prompt = f"Extract nodes and relationships from the following text:\n{raw_data}"

    parsed_response = ollama_llm_parser(prompt)  
    parsed_response = parsed_response.graph  

    nodes = {}
    relationships = []

    for entry in parsed_response:
        node = entry.node
        target_node = entry.target_node  
        relationship = entry.relationship  

        # Add nodes to the dictionary with a unique ID
        if node not in nodes:
            nodes[node] = str(uuid.uuid4())

        if target_node and target_node not in nodes:
            nodes[target_node] = str(uuid.uuid4())

        # Add relationship to the relationships list with node IDs
        if target_node and relationship:
            relationships.append({
                "source": nodes[node],
                "target": nodes[target_node],
                "type": relationship
            })

    return nodes, relationships

def ingest_to_neo4j(nodes, relationships):
    """
    Ingest nodes and relationships into Neo4j.
    """

    with neo4j_driver.session() as session:
        # Create nodes in Neo4j
        for name, node_id in nodes.items():
            session.run(
                "CREATE (n:Entity {id: $id, name: $name})",
                id=node_id,
                name=name
            )

        # Create relationships in Neo4j
        for relationship in relationships:
            session.run(
                "MATCH (a:Entity {id: $source_id}), (b:Entity {id: $target_id}) "
                "CREATE (a)-[:RELATIONSHIP {type: $type}]->(b)",
                source_id=relationship["source"],
                target_id=relationship["target"],
                type=relationship["type"]
            )

    return nodes

def create_collection(client, collection_name, vector_dimension):
    # Try to fetch the collection status
    try:
        collection_info = client.get_collection(collection_name)
        print(f"Skipping creating collection; '{collection_name}' already exists.")
    except Exception as e:
        # If collection does not exist, an error will be thrown, so we create the collection
        if 'Not found: Collection' in str(e):
            print(f"Collection '{collection_name}' not found. Creating it now...")

            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=vector_dimension, distance=models.Distance.COSINE)
            )

            print(f"Collection '{collection_name}' created successfully.")
        else:
            print(f"Error while checking collection: {e}")

def ollama_embeddings(text):
    response = ollama.embeddings(
        model="nomic-embed-text",
        prompt=text
    )
    return response['embedding']

def ingest_to_qdrant(collection_name, raw_data, node_id_mapping):
    # 1. Chunk the text safely
    chunks = text_splitter.split_text(raw_data)

    points = []

    for chunk in chunks:
        embedding = ollama_embeddings(chunk)

        points.append(
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "text": chunk
                }
            )
        )

    # 2. Upsert to Qdrant
    qdrant_client.upsert(
        collection_name=collection_name,
        points=points
    )

def retriever_search(neo4j_driver, qdrant_client, collection_name, query):
    retriever = QdrantNeo4jRetriever(
        driver=neo4j_driver,
        client=qdrant_client,
        collection_name=collection_name,
        id_property_external="id",
        id_property_neo4j="id",
    )

    results = retriever.search(query_vector=ollama_embeddings(query), top_k=5)
    
    return results

def fetch_related_graph(neo4j_client, entity_ids):
    query = """
    MATCH (e:Entity)-[r1]-(n1)-[r2]-(n2)
    WHERE e.id IN $entity_ids
    RETURN e, r1 as r, n1 as related, r2, n2
    UNION
    MATCH (e:Entity)-[r]-(related)
    WHERE e.id IN $entity_ids
    RETURN e, r, related, null as r2, null as n2
    """
    with neo4j_client.session() as session:
        result = session.run(query, entity_ids=entity_ids)
        subgraph = []
        for record in result:
            subgraph.append({
                "entity": record["e"],
                "relationship": record["r"],
                "related_node": record["related"]
            })
            if record["r2"] and record["n2"]:
                subgraph.append({
                    "entity": record["related"],
                    "relationship": record["r2"],
                    "related_node": record["n2"]
                })
    return subgraph


def format_graph_context(subgraph):
    nodes = set()
    edges = []

    for entry in subgraph:
        entity = entry["entity"]
        related = entry["related_node"]
        relationship = entry["relationship"]

        nodes.add(entity["name"])
        nodes.add(related["name"])

        edges.append(f"{entity['name']} {relationship['type']} {related['name']}")

    return {"nodes": list(nodes), "edges": edges}

def graphRAG_run(graph_context, user_query):
    nodes_str = ", ".join(graph_context["nodes"])
    edges_str = "; ".join(graph_context["edges"])
    prompt = f"""
    You are an intelligent assistant with access to the following knowledge graph:

    Nodes: {nodes_str}

    Edges: {edges_str}

    Using this graph, Answer the following question:

    User Query: "{user_query}"
    """
    
    try:
        response = ollama.chat(
            model="llama3.2",
            messages=[
                {"role": "system", "content": "Provide the answer for the following question:"},
                {"role": "user", "content": prompt}
            ]
        )
        return response['message']['content']
    
    except Exception as e:
        return f"Error querying LLM: {str(e)}"


def read_pdf_as_string(file_path: str) -> str:
    try:
        # Use hi_res strategy to identify tables if possible, otherwise falls back
        elements = partition(filename=file_path, languages=["eng"], strategy='hi_res')
        
        content_parts = []
        for el in elements:
            if el.category == "Table":
                # Ensure table_html is a string even if text_as_html is None
                table_html = getattr(el.metadata, "text_as_html", None) or str(el)
                content_parts.append(table_html)
            else:
                content_parts.append(str(el))
        
        raw_data = "\n\n".join(content_parts)
        return raw_data

    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
        return ""
    
# def read_document_as_paragraph_string(file_path: str) -> str:
#     try:
#         doc = Document(file_path)
#         full_text = []

#         for paragraph in doc.paragraphs:
#             if paragraph.text.strip():
#                 full_text.append(paragraph.text.strip())

#         raw_data = " ".join(full_text)
#         return raw_data

#     except Exception as e:
#         print(f"Error reading document {file_path}: {e}")
#         return ""

if __name__ == "__main__":
    print("Script started")
    print("Loading environment variables...")
    load_dotenv('.env.local')
    print("Environment variables loaded")
    
    print("Initializing clients...")
    # neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password))
    # qdrant_client = QdrantClient(
    #     url=qdrant_url,
    #     api_key=qdrant_key
    # )
    print("Clients initialized")
    
    print("Creating collection...")
    collection_name = "graphRAGstoreds_local"
    vector_dimension = 768 # nomic-embed-text dimension
    create_collection(qdrant_client, collection_name, vector_dimension)
    print("Collection created/verified")
    
    print("Extracting graph components...")
    
    file_path = os.path.join(os.path.dirname(__file__),"VTS", "G1110-Ed2.1-Use-of-Decision-Support-Tools-for-VTS-Personnel-January-2022.pdf")
    raw_data = read_pdf_as_string(file_path)
    # raw_data = read_document_as_paragraph_string(file_path)
    # Using a subset of data if it's too large for the LLM context in one go
    # For now, let's take the first 4000 characters to be safe with local llama
    raw_data_subset = raw_data[:4000] 
    print("RAW_D (subset): ", raw_data_subset)
    nodes, relationships = extract_graph_components(raw_data_subset)
    print("Nodes:", nodes)
    print("Relationships:", relationships)
    
    print("Ingesting to Neo4j...")
    node_id_mapping = ingest_to_neo4j(nodes, relationships)
    print("Neo4j ingestion complete")
    
    print("Ingesting to Qdrant...")
    ingest_to_qdrant(collection_name, raw_data_subset, node_id_mapping)
    print("Qdrant ingestion complete")

    query = "A VTS Authority wants to review the last year of traffic data to identify 'high-risk' areas where grounding is most likely to occur based on vessel size and type. Which Long term (Planning) DST is designed for this?"
    print("Starting retriever search...")
    retriever_result = retriever_search(neo4j_driver, qdrant_client, collection_name, query)
    print("Retriever results:", retriever_result)
    
    print("Extracting entity IDs...")
    entity_ids = [item.content.split("'id': '")[1].split("'")[0] for item in retriever_result.items]
    print("Entity IDs:", entity_ids)
    
    print("Fetching related graph...")
    subgraph = fetch_related_graph(neo4j_driver, entity_ids)
    print("Subgraph:", subgraph)
    
    print("Formatting graph context...")
    graph_context = format_graph_context(subgraph)
    print("Graph context:", graph_context)
    
    print("Running GraphRAG...")
    answer = graphRAG_run(graph_context, query)
    print("Final Answer:", answer)

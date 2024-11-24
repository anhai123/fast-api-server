from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from langchain_community.embeddings import OpenAIEmbeddings
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
import os

from dotenv import load_dotenv
load_dotenv()

open_ai_key = os.environ['OPENAI_API_KEY']
url_qdrant_client = os.environ['QDRANT_CLOUD_URL']  # Replace with your Qdrant Cloud URL
api_key_qdrant_client =os.environ['QDRANT_CLOUD_API_KEY']  # Replace with your Qdrant Cloud API key

# Initialize OpenAIEmbeddings with a 768-dimensional model
embedding_model = OpenAIEmbeddings(
    openai_api_key=open_ai_key,
    model="text-embedding-ada-002"  # Explicitly specify the model
)

# embedding_model = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-mpnet-base-v2"  # 768 dimensions
# )

# Initialize Qdrant client for Qdrant Cloud
qdrant_client = QdrantClient(
    url=url_qdrant_client,  # Replace with your Qdrant Cloud URL
    api_key=api_key_qdrant_client,  # Replace with your Qdrant Cloud API key
)


# Define your collection name
collection_name = "JobHunting"

# Connect LangChain with Qdrant
vector_store = Qdrant(
    client=qdrant_client,
    collection_name=collection_name,
    embeddings=embedding_model
)

# Ensure the collection exists; if not, create it
collections = [collection.name for collection in qdrant_client.get_collections().collections]
if collection_name not in collections:
    print(collections)
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "size": 1536,  # Update based on your embedding model
            "distance": 'Cosine'  # or 'Euclidean', 'Dot'
        }
    )

# qdrant_client.delete_collection(collection_name)

# qdrant_client.create_collection(
#         collection_name=collection_name,
#         vectors_config={
#             "size": 1536,  # Update based on your embedding model
#             "distance": 'Cosine'  # or 'Euclidean', 'Dot'
#         }
#     )

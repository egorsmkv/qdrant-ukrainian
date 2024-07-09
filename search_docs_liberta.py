"""
Code adapted from https://colab.research.google.com/drive/1Bz8RSVHwnNDaNtDwotfPj0w7AYzsdXZ-?usp=sharing#scrollTo=sXJ4rQNBhl4S
"""

from qdrant_client import QdrantClient
from qdrant_client.http.models import NamedVector
from sentence_transformers import SentenceTransformer

# Config
COLLECTION_NAME = "djinni_CVs"
QDRANT_ADDR = "http://localhost:6333"
QUERY_TEXT = "адоб фотошоп"

# Model to create embeddings
encoder = SentenceTransformer("Goader/liberta-large-v2", trust_remote_code=True)

# Create client
qdrant = QdrantClient(QDRANT_ADDR)

# Encode the query
encoded_vector = encoder.encode(QUERY_TEXT).tolist()

print("query:", QUERY_TEXT)
print("---")

# Let's now search
hits = qdrant.search(
    collection_name=COLLECTION_NAME,
    query_vector=NamedVector(
        name="cv",
        vector=encoded_vector,
    ),
    limit=5,
    with_vectors=False,
    with_payload=True,
)
for hit in hits:
    print(hit.payload, "score:", hit.score)
    print()

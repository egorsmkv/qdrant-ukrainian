"""
Code adapted from https://colab.research.google.com/drive/1Bz8RSVHwnNDaNtDwotfPj0w7AYzsdXZ-?usp=sharing#scrollTo=sXJ4rQNBhl4S
"""

import json

from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer

# Config
COLLECTION_NAME = "djinni_CVs"
QDRANT_ADDR = "http://localhost:6333"
QUERY_TEXT = "java розробник"

# Model to create embeddings
encoder = SentenceTransformer("lang-uk/ukr-paraphrase-multilingual-mpnet-base")

# Open our docs
with open("documents.json") as f:
    documents = json.load(f)
    print("Documents:", len(documents))

documents = documents[:300]

# Create client
qdrant = QdrantClient(QDRANT_ADDR)


# A util function to make batches
def make_batches(iterable, n=1):
    it_len = len(iterable)
    for ndx in range(0, it_len, n):
        yield iterable[ndx : min(ndx + n, it_len)]


# A helper function
def emptify_none(v):
    if not v:
        return ""
    return v


# Check collection
exists = qdrant.collection_exists(
    collection_name=COLLECTION_NAME,
)

if not exists:
    # Create collection
    dim = encoder.get_sentence_embedding_dimension()
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "position": models.VectorParams(size=dim, distance=models.Distance.COSINE),
            "cv": models.VectorParams(size=dim, distance=models.Distance.COSINE),
            "moreinfo": models.VectorParams(size=dim, distance=models.Distance.COSINE),
            "lookingfor": models.VectorParams(
                size=dim, distance=models.Distance.COSINE
            ),
            "highlights": models.VectorParams(
                size=dim, distance=models.Distance.COSINE
            ),
        },
    )

    # Let's vectorize CVs and upload to qdrant
    bs = 50
    left_docs = len(documents)
    for batch in make_batches(documents, bs):
        qdrant.upload_points(
            collection_name=COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=idx,
                    vector={
                        "position": encoder.encode(
                            emptify_none(doc["Position"])
                        ).tolist(),
                        "cv": encoder.encode(emptify_none(doc["CV"])).tolist(),
                        "moreinfo": encoder.encode(
                            emptify_none(doc["Moreinfo"])
                        ).tolist(),
                        "lookingfor": encoder.encode(
                            emptify_none(doc["Looking For"])
                        ).tolist(),
                        "highlights": encoder.encode(
                            emptify_none(doc["Highlights"])
                        ).tolist(),
                    },
                    payload=doc,
                )
                for idx, doc in enumerate(batch)
            ],
        )

        left_docs -= bs
        print("Uploaded batch of documents... Left:", left_docs)

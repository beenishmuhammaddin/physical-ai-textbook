# hackathon_qdrant.py
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# -------------------------
# 1) Connect to Qdrant Cloud
# -------------------------
client = QdrantClient(
    url="https://aca1129d-4d5e-4a41-8c11-7218f4c4302f.europe-west3-0.gcp.cloud.qdrant.io",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.q6nMxpGbdTiRZnT7acBbqzxyeW-F3EBy9ezngBJBgms"
)

collection_name = "hackathon_vectors"

# -------------------------
# 2) Create collection (only once)
# -------------------------
if not client.collection_exists(collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
    )
    print("✅ New collection created")
else:
    print("✅ Collection already exists")

# -------------------------
# 3) Insert sample vector
# -------------------------
client.upsert(
    collection_name=collection_name,
    points=[
        PointStruct(
            id=1,
            vector=[0.2] * 1536,
            payload={"text": "sample data"}
        )
    ]
)

print("✅ Sample vector inserted")

# -------------------------
# 4) Search using query_points (Correct for v1.16+)
# -------------------------
search_result = client.query_points(
    collection_name=collection_name,
    query=[0.1] * 1536,
    limit=1
)

print("✅ Search result:", search_result)

from qdrant_client import QdrantClient

API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.q6nMxpGbdTiRZnT7acBbqzxyeW-F3EBy9ezngBJBgms"
ENDPOINT = "https://aca1129d-4d5e-4a41-8c11-7218f4c4302f.europe-west3-0.gcp.cloud.qdrant.io"

client = QdrantClient(
    url=ENDPOINT,
    api_key=API_KEY
)

# Collections check karna
collections = client.get_collections()
print("Collections in your cluster:", collections)

# Nayi collection create karna (agar chahiye)
client.recreate_collection(
    collection_name="hackathon_collection",
    vector_size=1536,   # vector dimension, jo model ke hisaab se hoga
    distance="Cosine"   # ya "Euclid" ya "Dot"
)
print("New collection created: hackathon_collection")

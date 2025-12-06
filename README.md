# Hackathon Qdrant Vector Search

This project demonstrates using Qdrant Cloud to create a collection, insert a sample vector, and perform a search using Python. It is built for the hackathon and fully tested.

---

## Features

- Connects to Qdrant Cloud via API key
- Creates a collection (`hackathon_vectors`) if it doesn't exist
- Inserts a sample vector with payload
- Performs a vector search (query) and displays results

---

## Requirements

- Python 3.12+
- `qdrant-client` library

Install the library with:

```bash
pip install qdrant-client

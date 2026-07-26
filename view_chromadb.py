import chromadb

# Connect to the existing ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")

# List all collections
collections = client.list_collections()

print("=" * 60)
print("           CHROMADB DATABASE")
print("=" * 60)

for collection in collections:
    print(f"\nCollection Name : {collection.name}")

    col = client.get_collection(collection.name)

    data = col.get()

    print(f"Total Chunks    : {len(data['documents'])}")

    print("\nFirst Document Chunk:")
    print("-" * 60)
    print(data["documents"][0])

    print("\nMetadata:")
    print("-" * 60)
    print(data["metadatas"][0])

    print("\nChunk ID:")
    print("-" * 60)
    print(data["ids"][0])

print("\n" + "=" * 60)
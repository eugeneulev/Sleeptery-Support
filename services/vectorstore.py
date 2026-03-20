import chromadb


class VectorStore:
    def __init__(self, persist_dir: str):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="knowledge_base",
            metadata={"hnsw:space": "cosine"},
        )

    def add(
        self,
        documents: list[str],
        metadatas: list[dict],
        ids: list[str],
    ) -> None:
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )

    def query(self, query: str, n_results: int = 5) -> dict:
        return self.collection.query(query_texts=[query], n_results=n_results)

    def count(self) -> int:
        return self.collection.count()

import os


class Settings:
    DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gemini/gemini-2.0-flash")
    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_data")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    TOP_K = int(os.getenv("TOP_K", "5"))


settings = Settings()

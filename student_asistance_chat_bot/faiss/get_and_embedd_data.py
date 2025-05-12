import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from api_try import get_data_from_airtable, API_KEY, BASE_ID, TABLE_NAME

EMBEDDINGS_FILE = "embeddings.npy"
METADATA_FILE = "metadata.pkl"
FAISS_FILE = "faiss_index.index"

def build_faiss_index():
    embeddings = np.load(EMBEDDINGS_FILE)

    index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 mesafesi için düz bir indeks oluştur
    index.add(embeddings)

    # Kaydet
    faiss.write_index(index, FAISS_FILE)
    print(f"FAISS index oluşturuldu ve '{FAISS_FILE}' dosyasına kaydedildi.")

def main():
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    # Airtable'dan veri çek
    data = get_data_from_airtable(API_KEY, BASE_ID, TABLE_NAME)

    # Embedding için sadece soruları al
    sorular = [item["soru"] for item in data]
    embeddings = model.encode(sorular)

    # Embedding'lerle eşleşecek metadata: sadece id ve soru bilgisini sakla
    metadata = [
        {"id": item.get("id"), "soru": item.get("soru")}
        for item in data
    ]

    # Dosyalara kaydet
    np.save(EMBEDDINGS_FILE, embeddings)
    with open(METADATA_FILE, "wb") as f:
        pickle.dump(metadata, f)

    print(f"{len(sorular)} soru embedlendi ve metadata kaydedildi.")

if __name__ == "__main__":
    main()
    build_faiss_index()

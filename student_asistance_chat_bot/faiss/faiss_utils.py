import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from api_try import API_KEY, BASE_ID, TABLE_NAME
from api_try import get_data_from_airtable, get_answer_by_id  # get_answer_by_id fonksiyonunu sen yazacaksın

EMBEDDINGS_FILE = "embeddings.npy"
METADATA_FILE = "metadata.pkl"
FAISS_FILE = "faiss_index.index"
SIMILARITY_THRESHOLD = 0.8

def cosine_similarity(vec1, vec2):
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)

def process_user_question(user_question):
    # Gerekli dosyaları ve modeli yükle
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    query_embedding = model.encode([user_question]).astype("float32")

    index = faiss.read_index(FAISS_FILE)
    with open(METADATA_FILE, "rb") as f:
        metadata = pickle.load(f)

    # En yakın sonucu bul
    distances, indices = index.search(query_embedding, 1)
    matched_idx = indices[0][0]
    matched_embedding = np.load(EMBEDDINGS_FILE)[matched_idx]

    similarity = cosine_similarity(query_embedding[0], matched_embedding)
    print(f"Benzerlik: {similarity:.2f}")

    if similarity < SIMILARITY_THRESHOLD:
        print("Sorduğunuz soru veri tabanındaki sorularla yeterince benzer değil. Lütfen farklı bir şekilde sorun.")
    else:
        matched_id = metadata[matched_idx]["id"]
        # Bu fonksiyon Airtable'dan id'ye göre cevabı almalı (senin yazman gereken kısım)
        cevap = get_answer_by_id(API_KEY, BASE_ID, TABLE_NAME, matched_id)
        print(f"Cevap: {cevap}")

# Kullanıcıdan giriş al
if __name__ == "__main__":
    user_input = input("Bir soru sorun: ")
    process_user_question(user_input)

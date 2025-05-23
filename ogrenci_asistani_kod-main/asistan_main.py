import numpy as np
import faiss
import pickle
import time
import os
from sentence_transformers import SentenceTransformer
from api_try import *
from openai_integration import *
from dotenv import load_dotenv

load_dotenv("secrets.env")
API_KEY = os.getenv("AIRTABLE_API_KEY")
BASE_ID = os.getenv("BASE_ID")
TABLE_NAME = os.getenv("TABLE_NAME")
FAISS_FILE = "faiss_index.index"
SIMILARITY_THRESHOLD = 0.8
METADATA_FILE = "metadata.pkl"

## Benzerlik oranı ölçme
def cosine_similarity(vec1, vec2):
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def process_user_question(user_question):
    start = time.time()
    embedding = model.encode([user_question])[0]

    index = faiss.read_index(FAISS_FILE)
    D, I = index.search(np.array([embedding]).astype("float32"), k=1)
    similarity = 1 - D[0][0]

    if similarity > SIMILARITY_THRESHOLD:
        with open(METADATA_FILE, "rb") as f:
            metadata = pickle.load(f)
        matched = metadata[I[0][0]]
        answer = get_answer_by_id(API_KEY, BASE_ID, TABLE_NAME, matched["id"])
        end = time.time()
        return f"Benzer soru bulundu.\nCevap: {answer}\n(Süre: {end - start:.2f} saniye)"
    else:
        if is_technical_question_gpt(user_question):
            gpt_answer = get_gpt_answer(user_question)
            new_id = add_question_to_airtable(API_KEY, BASE_ID, TABLE_NAME, user_question, gpt_answer)
            if new_id:
                new_embedding = model.encode([user_question])[0]
                append_new_question_to_embeddings(user_question, new_embedding, {"id": new_id, "soru": user_question})
                end = time.time()
                return f"Benzer soru bulunamadı. GPT'den cevap alındı ve kaydedildi:\n{gpt_answer}\n(Süre: {end - start:.2f} saniye)"
            else:
                return "Cevap kaydedilirken hata oluştu."
        else:
            return "Bu sistem yalnızca teknik sorulara cevap verir."


if __name__ == "__main__":
    exit_flag = True

    while(exit_flag):
        user_input = input("Bir soru sorun: ")
        process_user_question(user_input)
        get_out = input("Devam etmek için enter, çıkış yapmak için 'e' tuşuna basınız.")
        if get_out.lower() == "e":
            print("Görüşmek üzere!")
            exit_flag = False
        else:
            continue

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import faiss
import pickle
import os
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import openai


# Ortam değişkenlerini yükle
load_dotenv("secrets.env")

# Airtable bilgileri
API_KEY = os.getenv("AIRTABLE_API_KEY")
BASE_ID = os.getenv("BASE_ID")
TABLE_NAME = os.getenv("TABLE_NAME")

# OpenAI API key ve client
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# Dosya yolları
EMBEDDINGS_FILE = "embeddings.npy"
METADATA_FILE = "metadata.pkl"
FAISS_FILE = "faiss_index.index"
SIMILARITY_THRESHOLD = 0.8

# FastAPI app
app = FastAPI()

# Model yükle (bir kere yüklensin)
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Kullanıcıdan gelecek veri şeması
class Question(BaseModel):
    question: str

# Helper fonksiyonlar (senin fonksiyonlarını burada toplayabiliriz)
def cosine_similarity(vec1, vec2):
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)

# Airtable erişim fonksiyonları (buraya senin requests kodlarını koy)
import requests

def get_answer_by_id(record_id):
    url = f"https://api.airtable.com/v0/{BASE_ID}/{TABLE_NAME}/{record_id}"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        record = response.json()
        return record.get("fields", {}).get("Cevap", None)
    else:
        return None

def add_question_to_airtable(soru, cevap):
    url = f"https://api.airtable.com/v0/{BASE_ID}/{TABLE_NAME}"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "fields": {
            "Soru": soru,
            "Cevap": cevap
        }
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code in (200, 201):
        return response.json().get("id")
    else:
        return None

# OpenAI teknik kontrol
def is_technical_question_gpt(question):
    system_prompt = (
        "Kullanıcının sorduğu sorunun yapay zeka, makine öğrenmesi, derin öğrenme "
        "veya veri bilimiyle ilgili olup olmadığını kontrol et. Eğer ilgiliyse 'EVET', değilse 'HAYIR' yanıtı ver."
    )
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
    )
    answer = response.choices[0].message.content.strip().upper()
    return answer == "EVET"

# OpenAI cevabı
def get_gpt_answer(question):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Makine öğrenmesi alanında uzman bir asistansın. Açık ve sade cevaplar ver."},
            {"role": "user", "content": question}
        ]
    )
    return response.choices[0].message.content.strip()

# Embedding ve FAISS güncelleme
def append_new_question_to_embeddings(question, embedding, metadata_entry):
    # Embeddings yükle, ekle, kaydet
    embeddings = np.load(EMBEDDINGS_FILE)
    embeddings = np.vstack([embeddings, embedding])
    np.save(EMBEDDINGS_FILE, embeddings)

    # Metadata yükle, ekle, kaydet
    with open(METADATA_FILE, "rb") as f:
        metadata = pickle.load(f)
    metadata.append(metadata_entry)
    with open(METADATA_FILE, "wb") as f:
        pickle.dump(metadata, f)

    # FAISS index güncelle
    index = faiss.read_index(FAISS_FILE)
    index.add(np.array([embedding]).astype("float32"))
    faiss.write_index(index, FAISS_FILE)

# Ana endpoint
@app.post("/ask")
async def ask_question(q: Question):
    user_question = q.question

    # 1. Embedding hesapla
    embedding = model.encode([user_question])[0].astype("float32")

    # 2. FAISS index oku ve arama yap
    if not os.path.exists(FAISS_FILE) or not os.path.exists(METADATA_FILE):
        raise HTTPException(status_code=500, detail="FAISS index veya metadata dosyası bulunamadı. Lütfen index oluşturun.")

    index = faiss.read_index(FAISS_FILE)
    D, I = index.search(np.array([embedding]), k=1)
    similarity = 1 - D[0][0]  # D matrisi L2 mesafesi, benzerlik dönüşümü

    with open(METADATA_FILE, "rb") as f:
        metadata = pickle.load(f)

    if similarity > SIMILARITY_THRESHOLD:
        matched = metadata[I[0][0]]
        answer = get_answer_by_id(matched["id"])
        if answer:
            return {"answer": answer}
        else:
            return {"answer": "Benzer soru bulundu fakat cevap alınamadı."}

    # 3. Teknik soru mu diye kontrol et
    if not is_technical_question_gpt(user_question):
        return {"answer": "Bu sistem yalnızca teknik sorulara cevap verir."}

    # 4. Teknik ise OpenAI cevabı al
    answer = get_gpt_answer(user_question)

    # 5. Airtable ve index güncelle (async olabilir, burada sync yapıyoruz)
    new_id = add_question_to_airtable(user_question, answer)
    if new_id:
        append_new_question_to_embeddings(user_question, embedding, {"id": new_id, "soru": user_question})

    return {"answer": answer}

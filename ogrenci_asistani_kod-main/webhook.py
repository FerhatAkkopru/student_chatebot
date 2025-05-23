from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import numpy as np
import time
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from api_try import *
from openai_integration import *
import os
from dotenv import load_dotenv
import requests

app = FastAPI()

# .env dosyasını yükle
load_dotenv("secrets.env") # Proje kök dizininde secrets.env dosyası olduğundan emin olun
API_KEY = os.getenv("AIRTABLE_API_KEY")
BASE_ID = os.getenv("BASE_ID")
TABLE_NAME = os.getenv("TABLE_NAME")

# Dosya yolları ve sabitler
EMBEDDINGS_FILE = "embeddings.npy" # Bu dosya şu anki mantıkta doğrudan kullanılmıyor ama referans olarak kalabilir
METADATA_FILE = "metadata.pkl"
FAISS_FILE = "faiss_index.index"
SIMILARITY_THRESHOLD = 0.8 # Benzerlik eşiği

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
    embedding = np.array([embedding]).astype("float32")
    faiss.normalize_L2(embedding)  # Şimdi gerçekten normalize oldu

    index = faiss.read_index(FAISS_FILE)
    D, I = index.search(embedding, k=1)

    similarity = D[0][0]  # Bu artık doğru cosine similarity değeri


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

@app.post("/webhook")
async def webhook(request: Request):
    try:
        body = await request.json()
        question = body.get("question")

        if not question:
            return JSONResponse(content={"enswwer":"error No question provided."}, status_code=400)

        if not is_technical_question_gpt(question):
            return JSONResponse(content={"answer":"Bu sistem yalnızca teknik sorulara cevap verir."})

        answer = process_user_question(question)
        return JSONResponse(content={"answer":answer})

    except Exception as e:
        print(f"Error in webhook: {str(e)}")
        return JSONResponse(content={"answer":f"error: {str(e)}"}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    # Uvicorn'u çalıştırırken reload özelliğini geliştirme aşamasında kullanabilirsiniz.
    uvicorn.run("webhook:app", host="0.0.0.0", port=8000, reload=True)

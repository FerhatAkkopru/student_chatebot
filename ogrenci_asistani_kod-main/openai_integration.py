import openai
import numpy as np
import pickle
import faiss
from dotenv import load_dotenv
import os

load_dotenv("secrets.env")

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def is_technical_question_gpt(question):
    system_prompt = (
        "Kullanıcının sorduğu sorunun yapay zeka, makine öğrenmesi, derin öğrenme "
        "veya veri bilimiyle ilgili olup olmadığını kontrol et. Eğer ilgiliyse 'EVET', değilse 'HAYIR' yanıtı ver."
    )

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
    )

    answer = response.choices[0].message.content.strip().upper()
    return answer == "EVET"

def get_gpt_answer(question):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Makine öğrenmesi alanında uzman bir asistansın. Açık ve sade cevaplar ver."},
            {"role": "user", "content": question}
        ]
    )
    return response.choices[0].message.content.strip()

def append_new_question_to_embeddings(question, embedding, metadata_entry):
    embeddings = np.load("embeddings.npy")
    embeddings = np.vstack([embeddings, embedding])
    np.save("embeddings.npy", embeddings)

    with open("metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    metadata.append(metadata_entry)
    with open("metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    # FAISS index güncelle
    index = faiss.read_index("faiss_index.index")
    index.add(np.array([embedding]).astype("float32"))
    faiss.write_index(index, "faiss_index.index")

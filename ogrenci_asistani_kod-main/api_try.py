import requests
import os
from dotenv import load_dotenv

load_dotenv("secrets.env")

API_KEY = os.getenv("AIRTABLE_API_KEY")
BASE_ID = os.getenv("BASE_ID")
TABLE_NAME = os.getenv("TABLE_NAME")

def get_data_from_airtable(api_key, base_id, table_name):
    url = f"https://api.airtable.com/v0/{base_id}/{table_name}"
    headers = {
        "Authorization": f"Bearer {api_key}",
    }
    response = requests.get(url, headers=headers)
    records = response.json()["records"]
    
    data = []
    for record in records:
        fields = record["fields"]
        if "Soru" in fields and "Cevap" in fields:
            data.append({
                "id": record["id"],
                "soru": fields["Soru"],
                "cevap": fields["Cevap"]
            })
    return data

def get_answer_by_id(api_key, base_id, table_name, record_id):
    url = f"https://api.airtable.com/v0/{base_id}/{table_name}/{record_id}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        record = response.json()
        fields = record.get("fields", {})
        return fields.get("Cevap", None)
    else:
        print(f"Hata: {response.status_code} - {response.text}")
        return None


def add_question_to_airtable(api_key, base_id, table_name, soru, cevap):
    url = f"https://api.airtable.com/v0/{base_id}/{table_name}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "fields": {
            "Soru": soru,
            "Cevap": cevap
        }
    }

    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200 or response.status_code == 201:
        return response.json()["id"]
    else:
        print(f"Hata (kayıt ekleme): {response.status_code} - {response.text}")
        return None


veriler = get_data_from_airtable(API_KEY, BASE_ID, TABLE_NAME)


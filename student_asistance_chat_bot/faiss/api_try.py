import requests

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

API_KEY = "pat6HZFg71xnlKZmg.48f68492e7191dd5617dfd1f097fd35f9ab416f2ec07af1fb0c5f7daa8e2ad15"
BASE_ID = "appk4emRuA6vIrCMu"
TABLE_NAME = "Data"

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

veriler = get_data_from_airtable(API_KEY, BASE_ID, TABLE_NAME)


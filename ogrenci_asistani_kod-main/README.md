# Akıllı Soru-Cevap Sistemi (Makine Öğrenmesi Odaklı)

Bu proje, yapay zeka ve makine öğrenmesi alanındaki soruları anlayıp anlamlı cevaplar verebilen bir Python tabanlı sistemdir. Teknik olmayan sorular ise GPT modeli kullanılarak filtrelenir veya dışlanır.

---

## 🔍 Proje Amacı

- Kullanıcının sorduğu soruyu vektör uzayında temsil ederek daha önce kayıtlı sorularla benzerlik karşılaştırması yapmak.
- Yeterli benzerlik varsa ilgili cevabı veritabanından çekmek.
- Teknik olmayan ya da sistemde benzeri bulunmayan sorularda OpenAI API'si (GPT) ile dinamik cevap üretmek.
- Gerekirse yeni soru-cevapları veri kümesine ve FAISS index’ine eklemek.

---

## 🧠 Temel Özellikler

- **SentenceTransformer ile gömleme (embedding)** oluşturma
- **FAISS** ile benzerlik karşılaştırması
- **Airtable API** üzerinden veri çekme ve cevap alma
- **OpenAI GPT-3.5 Turbo** ile teknik cevap üretimi
- Yeni soruların embedding'lerinin kayıt altına alınması ve index'e dahil edilmesi

---

## 📁 Proje Yapısı

- `main.py`: Kullanıcı etkileşimi, embedding karşılaştırması ve cevap akışı
- `embedding_tools.py`: Gömleme işlemleri ve FAISS index yönetimi
- `api_try.py`: Airtable API işlemleri
- `secrets.env`: API anahtarlarının saklandığı gizli dosya (versiyon kontrolüne dahil edilmez)
- `metadata.pkl`, `embeddings.npy`, `faiss_index.index`: Embedding ve index dosyaları

---

## 📦 Gereksinimler

- Python 3.7+
- `sentence-transformers`
- `faiss`
- `numpy`
- `requests`
- `openai`
- `python-dotenv`

---

## 🔐 Güvenlik Notu

API anahtarları ve hassas bilgiler `secrets.env` dosyasında saklanmalıdır. Bu dosya `.gitignore` dosyası ile versiyon kontrolüne dahil edilmemelidir.

---

## 🛠 Kurulum Adımları

1. Gerekli Python paketlerini yükleyin.
2. `secrets.env` dosyasını oluşturup API anahtarlarınızı girin.
3. Airtable verisini çekin ve embedding dosyalarını oluşturun.
4. `main.py` dosyasını çalıştırarak kullanıcı etkileşimini başlatın.

---

## 📌 Notlar

- Sistem sadece makine öğrenmesi ve ilgili konulara odaklanır.
- Yetersiz benzerlik veya alakasız sorular GPT’ye yönlendirilir ya da reddedilir.
- Yeni sorular anlamlıysa embedding sistemine eklenerek modelin kapsamı genişletilir.

---

## 🧑‍💻 Katkıda Bulunmak

Pull request’ler veya öneriler her zaman memnuniyetle karşılanır. Lütfen katkı sağlamadan önce yapıyı inceleyin.

---

## 👥 Yazarlar

Bu proje, öncülüğünü **Ferhat Akköprü**'nün yaptığı ve **Özlem Nur Duman**'ın katkılarıyla geliştirilen bir çalışmadır.

import streamlit as st
from asistan_main import process_user_question  # process_user_question fonksiyonunu bu dosyadan alıyoruz

# Sayfa ayarları
st.set_page_config(page_title="Öğrenci Asistan Chat Bot", page_icon="🧠", layout="centered")

# Sol üst köşeye küçük yazı ekle (sidebar olmadan)
st.markdown(
    """
    <div style='position: absolute; top: 10px; left: 10px; font-size: 14px; color: gray;'>
        Özllm Ltd. Şti.
    </div>
    """,
    unsafe_allow_html=True
)

# Ana başlık
st.markdown("<h1 style='text-align: center; color: #FF4500;'>Öğrenci Asistanı Chat Botu</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Makine öğrenmesi ve veri bilimiyle ilgili tüm sorularınız için.</p>", unsafe_allow_html=True)

# Giriş kutusu
st.markdown("#### Soru Sor")
user_question = st.text_input("")

# Soru işleme
if st.button("Gönder"):
    if user_question.strip() == "":
        st.warning("Lütfen bir soru girin.")
    else:
        with st.spinner("Soru işleniyor..."):
            answer = process_user_question(user_question)
        st.markdown(
    f"""
    <div style='padding: 10px; border: 1px solid #ddd; border-radius: 10px; background-color: rgba(255, 255, 255, 0.1);'>
        <strong>Cevap:</strong><br>{answer}
    </div>
    """,
    unsafe_allow_html=True
)
        st.write(answer)

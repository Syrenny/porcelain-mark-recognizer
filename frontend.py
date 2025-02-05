import os
import streamlit as st
import requests
from backend.search import Fields

from environment import settings


st.title("Определитель марок фарфоровых изделий")

uploaded_file = st.file_uploader(
    "Загрузите изображение марки", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Загруженное изображение", width=250)
    if st.button("Определить марку"):
        files = {"file": uploaded_file.getvalue()}
        response = requests.post(f"http://{settings.api_host}:{settings.api_port}/predict", files=files)

        if response.status_code == 200:
            results = response.json()
            st.subheader("Наиболее похожие марки:")
            for result in results:
                st.write(f"**Название:** {result[Fields.producer]}")
                # st.write(f"**Описание:** {result['description']}")
                st.image(result[Fields.path])
                st.markdown("---")
        else:
            st.error("Ошибка при обработке изображения. Попробуйте снова.")
    

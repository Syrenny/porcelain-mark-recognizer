import streamlit as st
import requests

st.title("Определитель марок фарфоровых изделий")

uploaded_file = st.file_uploader(
    "Загрузите изображение марки", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Загруженное изображение", width=250)
    if st.button("Определить марку"):
        files = {"file": uploaded_file.getvalue()}
        response = requests.post("http://localhost:8000/predict", files=files)

        if response.status_code == 200:
            results = response.json()["predictions"]
            st.subheader("Наиболее похожие марки:")
            for result in results:
                st.write(f"**Название:** {result['name']}")
                st.write(f"**Описание:** {result['description']}")
                st.image(result['image_url'])
                st.markdown("---")
        else:
            st.error("Ошибка при обработке изображения. Попробуйте снова.")

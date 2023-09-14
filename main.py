import streamlit as st
from tensorflow import keras
from PIL import Image
from utils import classify

# set title
st.title("Распознание наличия рака кожи по снимку опухоли")

# set header
st.header("Загрузите пожалуйста снимок участка кожи. Результат анализа отобразится снизу")

# file uploading
file = st.file_uploader("", type=["jpeg", "png", "jpg"])

# load classifier
model = keras.models.load_model("model/skin-cancer-detection-model.keras")

# display image
if file:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    # classify image
    confidence_score = classify(image=image, model=model)

    # write classification
    percentage = confidence_score * 100

    if percentage > 90.0:
        st.write(f"### С вероятностью в {percentage:.2f}% данный участок кожи поражен злокачественной меланомой")
    elif percentage < 10.0:
        st.write(f"### С вероятностью в {percentage:.2f}% на данном участке кожи нет злокачественной меланомы")
    else:
        st.write("### По данному снимку невозможно определить наличие опухоли либо изображение не содержит снимок МРТ")

st.write("Модель классификации использует данные со следующих источников:")
st.write("https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images")

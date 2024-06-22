import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Загрузка предобученной модели
model = tf.keras.models.load_model('mnist_cnn_model.h5')

# Функция для предобработки загруженного изображения
def preprocess_image(image):
    img = Image.open(image)
    img = img.convert('L')  # Преобразование в оттенки серого
    img = img.resize((28, 28))  # Изменение размера до 28x28
    img_array = np.array(img) / 255.0  # Преобразование в массив и нормализация
    img_array = img_array.reshape((1, 28, 28, 1))  # Изменение формы для модели CNN
    return img_array

# Заголовок и описание приложения
st.title('Распознавание цифр MNIST')
st.write('Нарисуйте цифру внизу на холсте, затем нажмите кнопку "Предсказать".')

# Виджет для рисования на холсте
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Цвет заливки с некоторой прозрачностью
    stroke_width=12,  # Ширина контура рисования
    stroke_color="black",  # Цвет контура
    background_color="white",  # Цвет фона холста
    update_streamlit=True,  # Обновление холста в реальном времени
    height=150,  # Высота холста
    drawing_mode="freedraw",  # Режим рисования - свободное рисование
    key="canvas",
)

# Проверяем, если ли данные изображения на холсте
if canvas_result.image_data is not None:
    st.image(canvas_result.image_data)

# Виджет для загрузки изображения для предсказания
uploaded_image = st.file_uploader("Загрузите изображение цифры (формат MNIST)", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img, caption='Загруженное изображение (Измененный размер)', use_column_width=True)

    with col2:
        st.write("")
        if st.button('Предсказать', key='classify_btn'):
            try:
                # Предобработка загруженного изображения
                img_array = preprocess_image(uploaded_image)

                # Предсказание с использованием предобученной модели
                result = model.predict(img_array)
                predicted_class = np.argmax(result)

                st.success(f'Предсказанная цифра: {predicted_class}')
            except Exception as e:
                st.error(f'Ошибка: {e}')

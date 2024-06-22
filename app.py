import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# Задаем параметры холста
drawing_mode = "freedraw"
stroke_width = 12
stroke_color = "black"
bg_color = st.sidebar.color_picker("Цвет фона:", "#eee")
bg_image = st.sidebar.file_uploader("Фоновое изображение:", type=["png", "jpg"])
realtime_update = st.sidebar.checkbox("Обновление в реальном времени", True)

# Создаем компонент холста
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Цвет заливки с некоторой прозрачностью
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    background_image=Image.open(bg_image) if bg_image else None,
    update_streamlit=realtime_update,
    height=150,
    drawing_mode=drawing_mode,
    key="canvas",
)

# Отображаем нарисованное изображение
if canvas_result.image_data is not None:
    st.image(canvas_result.image_data)

# Кнопка для сохранения нарисованного изображения
if st.button("Сохранить изображение"):
    drawn_image = Image.fromarray((canvas_result.image_data * 255).astype('uint8'))
    drawn_image.save("нарисованное_изображение.png")
    st.success("Изображение успешно сохранено!")

# Загрузка предварительно обученной модели
model = tf.keras.models.load_model('mnist_cnn_model.h5')

# Функция для предобработки загруженного изображения
def preprocess_image(image):
    img = Image.open(image)
    img = img.convert('L')  # Преобразование в оттенки серого
    img = img.resize((28, 28))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape((1, 28, 28, 1))
    return img_array

# Streamlit App
st.title('MNIST Digit Classifier')

# Функция для предобработки изображения с холста
def preprocess_canvas_image(image_data):
    img = Image.fromarray((image_data * 255).astype('uint8'))
    img = img.convert('L')  # Преобразование в оттенки серого
    img = img.resize((28, 28))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape((1, 28, 28, 1))
    return img_array

if st.button('Классифицировать нарисованную цифру'):
    try:
        # Предобработка изображения с холста
        img_array = preprocess_canvas_image(canvas_result.image_data)

        # Предсказание с использованием предварительно обученной модели
        result = model.predict(img_array)
        predicted_class = np.argmax(result)

        st.success(f'Предсказанная цифра: {predicted_class}')
    except Exception as e:
        st.error(f'Ошибка: {e}')

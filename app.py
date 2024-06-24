
############################################################################################

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt

import streamlit as st

# Устанавливаем стиль для центрирования элементов
st.markdown("""
    <style>
    .center {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100vh;
        flex-direction: column;
    }
    .center h1, .center p {
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Центрируем заголовок и текст
st.markdown('<div class="center">', unsafe_allow_html=True)
st.markdown('<h1>Классификатор рукописных цифр</h1>', unsafe_allow_html=True)
st.markdown('<p>Напишите одну цифру</p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)


# Streamlit App
# st.title('Классификатор рукописных цифр') #MNIST Digit Classifier
# st.write("Напишите одну цифру")

# Задаем параметры холста
drawing_mode = "freedraw"
stroke_width = 14
stroke_color = "black"  # Устанавливаем черный цвет для контура
bg_color = "white"  # Устанавливаем белый цвет фона
# bg_image = st.sidebar.file_uploader("Фоновое изображение:", type=["png", "jpg"])
# realtime_update = st.sidebar.checkbox("Обновление в реальном времени", True)

# Создаем компонент квадратной канвы
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Цвет заливки с некоторой прозрачностью
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    #background_image=Image.open(bg_image) if bg_image else None,
    # update_streamlit=realtime_update,
    height=150,  # Установим высоту и ширину канвы одинаковыми (в пикселях)
    width=150,
    drawing_mode=drawing_mode,
    key="canvas",
)

# Отображаем нарисованное изображение
# if canvas_result.image_data is not None:
#     st.image(canvas_result.image_data)

# Загрузка предварительно обученной модели
#model = tf.keras.models.load_model('mnist_cnn_model.h5')
#model = tf.keras.models.load_model('mnist_cnn_model.h5', compile=False)

# Функция для загрузки модели с дополнительными проверками
def load_model_safe(model_path):
    try:
        with tf.keras.backend.name_scope('model_loading'):
            model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except IndexError as e:
        print(f"IndexError: {e}. Возможно, стек областей видимости пуст.")
        # Дополнительная обработка или попытка повторной загрузки модели
        return None
    except Exception as e:
        print(f"Error loading the model: {e}")
        return None

# Попытка загрузки модели
model = load_model_safe('mnist_cnn_model.h5')

if model is not None:
    print("Model loaded successfully.")
else:
    print("Failed to load the model.")


##############################
def preprocess_canvas_image(image_data):
    img = Image.fromarray(image_data)
    img = img.convert('L')  # Преобразование в оттенки серого
    resized_image = img.resize((28, 28), Image.BILINEAR)  # Изменение размера
    grayscaled_image = resized_image.convert("L")  # Преобразование в grayscale
    inverted_image = 255 - np.array(grayscaled_image)  # Инверсия цветов
    normalized_image = inverted_image / 255.0  # Нормализация значений
    reshaped_image = normalized_image.reshape((1, 28, 28, 1))  # Изменение формы для модели CNN

    # Отображение предобработанных изображений с помощью Matplotlib
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(10, 6))

    # Скрытие меток осей для всех подграфиков
    for axis in ax.flat:
        axis.set_xticks([])
        axis.set_yticks([])

    # Исходное изображение
    ax[0, 0].set_title("Original Image")
    ax[0, 0].imshow(img, cmap='gray')

    # Измененное изображение (28 * 28)
    ax[0, 1].set_title("Resized Image")
    ax[0, 1].imshow(resized_image, cmap='gray')

    # Grayscale изображение
    ax[0, 2].set_title("Grayscale Image")
    ax[0, 2].imshow(grayscaled_image, cmap="gray")

    # Инвертированное изображение (текст белый, фон черный)
    ax[1, 0].set_title("Inverted Image")
    ax[1, 0].imshow(inverted_image, cmap="gray")

    # Нормализованное изображение (делим на 255, чтобы значения были от 0 до 1)
    ax[1, 1].set_title("Normalized Image")
    ax[1, 1].imshow(normalized_image, cmap="gray")

    # Измененная форма изображения
    ax[1, 2].set_title("Reshaped Image")
    ax[1, 2].imshow(reshaped_image.reshape((28, 28)), cmap="gray")

    st.pyplot(fig)

    return reshaped_image



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




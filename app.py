# import streamlit as st
# import numpy as np
# import tensorflow as tf
# from PIL import Image
# from streamlit_drawable_canvas import st_canvas

# # Задаем параметры холста
# drawing_mode = "freedraw"
# stroke_width = 12
# stroke_color = "black"  # Устанавливаем белый цвет для контура
# bg_color = "white"  # Устанавливаем черный цвет фона
# bg_image = st.sidebar.file_uploader("Фоновое изображение:", type=["png", "jpg"])
# realtime_update = st.sidebar.checkbox("Обновление в реальном времени", True)

# # Создаем компонент квадратной канвы
# canvas_result = st_canvas(
#     fill_color="rgba(255, 165, 0, 0.3)",  # Цвет заливки с некоторой прозрачностью
#     stroke_width=stroke_width,
#     stroke_color=stroke_color,
#     background_color=bg_color,
#     background_image=Image.open(bg_image) if bg_image else None,
#     update_streamlit=realtime_update,
#     height=150,  # Установим высоту и ширину канвы одинаковыми (в пикселях)
#     width=150,
#     drawing_mode=drawing_mode,
#     key="canvas",
# )

# # Отображаем нарисованное изображение
# if canvas_result.image_data is not None:
#     st.image(canvas_result.image_data)

# # Кнопка для сохранения нарисованного изображения
# if st.button("Сохранить изображение"):
#     drawn_image = Image.fromarray((canvas_result.image_data * 255).astype('uint8'))
#     drawn_image.save("нарисованное_изображение.png")
#     st.success("Изображение успешно сохранено!")

# # Загрузка предварительно обученной модели
# model = tf.keras.models.load_model('mnist_cnn_model.h5')

# # Функция для предобработки загруженного изображения
# def preprocess_image(image):
#     img = Image.open(image)
#     img = img.convert('L')  # Преобразование в оттенки серого
#     img = img.resize((28, 28))
#     img_array = np.array(img) / 255.0
#     img_array = img_array.reshape((1, 28, 28, 1))
#     return img_array

# # Streamlit App
# st.title('MNIST Digit Classifier')

# # Функция для предобработки изображения с холста
# def preprocess_canvas_image(image_data):
#     img = Image.fromarray((image_data * 255).astype('uint8'))
#     img = img.convert('L')  # Преобразование в оттенки серого
#     img = img.resize((28, 28))
#     img_array = np.array(img) / 255.0
#     img_array = img_array.reshape((1, 28, 28, 1))
#     return img_array

# if st.button('Классифицировать нарисованную цифру'):
#     try:
#         # Предобработка изображения с холста
#         img_array = preprocess_canvas_image(canvas_result.image_data)

#         # Предсказание с использованием предварительно обученной модели
#         result = model.predict(img_array)
#         predicted_class = np.argmax(result)

#         st.success(f'Предсказанная цифра: {predicted_class}')
#     except Exception as e:
#         st.error(f'Ошибка: {e}')

# import streamlit as st
# import numpy as np
# import tensorflow as tf
# from PIL import Image
# from streamlit_drawable_canvas import st_canvas
# import matplotlib.pyplot as plt

# # Задаем параметры холста
# drawing_mode = "freedraw"
# stroke_width = 15
# stroke_color = "black"  # Устанавливаем черный цвет для контура
# bg_color = "white"  # Устанавливаем белый цвет фона
# bg_image = st.sidebar.file_uploader("Фоновое изображение:", type=["png", "jpg"])
# realtime_update = st.sidebar.checkbox("Обновление в реальном времени", True)

# # Создаем компонент квадратной канвы
# canvas_result = st_canvas(
#     fill_color="rgba(255, 165, 0, 0.3)",  # Цвет заливки с некоторой прозрачностью
#     stroke_width=stroke_width,
#     stroke_color=stroke_color,
#     background_color=bg_color,
#     background_image=Image.open(bg_image) if bg_image else None,
#     update_streamlit=realtime_update,
#     height=150,  # Установим высоту и ширину канвы одинаковыми (в пикселях)
#     width=150,
#     drawing_mode=drawing_mode,
#     key="canvas",
# )

# # Отображаем нарисованное изображение
# if canvas_result.image_data is not None:
#     st.image(canvas_result.image_data)

# # Загрузка предварительно обученной модели
# model = tf.keras.models.load_model('mnist_cnn_model.h5')

# # Функция для предобработки изображения с холста
# def preprocess_canvas_image(image_data):
#     img = Image.fromarray((image_data * 255).astype('uint8'))

#     # Предобработка изображения
#     fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(10, 6))

#     # Скрытие меток осей для всех подграфиков
#     for axis in ax.flat:
#         axis.set_xticks([])
#         axis.set_yticks([])

#     # Исходное изображение
#     ax[0, 0].set_title("Original Image")
#     ax[0, 0].imshow(img, cmap='gray')

#     # Измененное изображение (28 * 28)
#     resized_image = img.resize((28, 28), Image.Resampling.BILINEAR)
#     ax[0, 1].set_title("Resized Image")
#     ax[0, 1].imshow(resized_image, cmap='gray')

#     # Grayscale изображение
#     grayscaled_image = resized_image.convert("L")
#     ax[0, 2].set_title("Grayscale Image")
#     ax[0, 2].imshow(grayscaled_image, cmap="gray")

#     # Инвертированное изображение (текст белый, фон черный)
#     inverted_image = 255 - np.array(grayscaled_image)
#     ax[1, 0].set_title("Inverted Image")
#     ax[1, 0].imshow(inverted_image, cmap="gray")

#     # Нормализованное изображение (делим на 255, чтобы значения были от 0 до 1)
#     normalized_image = inverted_image / 255.0
#     ax[1, 1].set_title("Normalized Image")
#     ax[1, 1].imshow(normalized_image, cmap="gray")

#     # Измененная форма изображения
#     reshaped_image = normalized_image.reshape((28, 28))
#     ax[1, 2].set_title("Reshaped Image")
#     ax[1, 2].imshow(reshaped_image, cmap="gray")

#     st.pyplot(fig)

#     return reshaped_image.reshape((1, 28, 28, 1))

# # Streamlit App
# st.title('MNIST Digit Classifier')

# if st.button('Классифицировать нарисованную цифру'):
#     try:
#         # Предобработка изображения с холста
#         img_array = preprocess_canvas_image(canvas_result.image_data)

#         # Предсказание с использованием предварительно обученной модели
#         result = model.predict(img_array)
#         predicted_class = np.argmax(result)

#         st.success(f'Предсказанная цифра: {predicted_class}')
#     except Exception as e:
#         st.error(f'Ошибка: {e}')

#########################################################################
# import streamlit as st
# import numpy as np
# import tensorflow as tf
# from PIL import Image
# from streamlit_drawable_canvas import st_canvas
# import matplotlib.pyplot as plt

# # Streamlit App
# st.title('Классификатор рукописных цифр MNIST')
# st.write("Протестируй работу MNIST!")
# st.write("Напиши одну цифру")

# # Задаем параметры холста
# drawing_mode = "freedraw"
# stroke_width = 12
# stroke_color = "black"  # Устанавливаем черный цвет для контура
# bg_color = "white"  # Устанавливаем белый цвет фона
# #bg_image = st.sidebar.file_uploader("Фоновое изображение:", type=["png", "jpg"])
# #realtime_update = st.sidebar.checkbox("Обновление в реальном времени", True)

# # Создаем компонент квадратной канвы
# canvas_result = st_canvas(
#     fill_color="rgba(255, 165, 0, 0.3)",  # Цвет заливки с некоторой прозрачностью
#     stroke_width=stroke_width,
#     stroke_color=stroke_color,
#     background_color=bg_color,
#     #background_image=Image.open(bg_image) if bg_image else None,
#     #update_streamlit=realtime_update,
#     height=150,  # Установим высоту и ширину канвы одинаковыми (в пикселях)
#     width=150,
#     drawing_mode=drawing_mode,
#     key="canvas",
# )

# # Отображаем нарисованное изображение
# if canvas_result.image_data is not None:
#     st.image(canvas_result.image_data)

# # Загрузка предварительно обученной модели
# model = tf.keras.models.load_model('mnist_cnn_model.h5')



# def preprocess_canvas_image(image_data):
#     img = Image.fromarray(image_data)
#     img = img.convert('L')  # Преобразование в оттенки серого
#     resized_image = img.resize((28, 28), Image.BILINEAR)  # Изменение размера
#     grayscaled_image = resized_image.convert("L")  # Преобразование в grayscale
#     inverted_image = 255 - np.array(grayscaled_image)  # Инверсия цветов
#     normalized_image = inverted_image / 255.0  # Нормализация значений
#     reshaped_image = normalized_image.reshape((1, 28, 28, 1))  # Изменение формы для модели CNN

#     # Отображение предобработанных изображений с помощью Matplotlib
#     fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(10, 6))

#     # Скрытие меток осей для всех подграфиков
#     for axis in ax.flat:
#         axis.set_xticks([])
#         axis.set_yticks([])

#     # Исходное изображение
#     ax[0, 0].set_title("Original Image")
#     ax[0, 0].imshow(img, cmap='gray')

#     # Измененное изображение (28 * 28)
#     ax[0, 1].set_title("Resized Image")
#     ax[0, 1].imshow(resized_image, cmap='gray')

#     # Grayscale изображение
#     ax[0, 2].set_title("Grayscale Image")
#     ax[0, 2].imshow(grayscaled_image, cmap="gray")

#     # Инвертированное изображение (текст белый, фон черный)
#     ax[1, 0].set_title("Inverted Image")
#     ax[1, 0].imshow(inverted_image, cmap="gray")

#     # Нормализованное изображение (делим на 255, чтобы значения были от 0 до 1)
#     ax[1, 1].set_title("Normalized Image")
#     ax[1, 1].imshow(normalized_image, cmap="gray")

#     # Измененная форма изображения
#     ax[1, 2].set_title("Reshaped Image")
#     ax[1, 2].imshow(reshaped_image.reshape((28, 28)), cmap="gray")

#     plt.tight_layout()
#     plt.show()

#     return reshaped_image


# # Streamlit App
# #st.title('MNIST Digit Classifier')

# if st.button('Классифицировать нарисованную цифру'):
#     try:
#         # Предобработка изображения с холста
#         img_array = preprocess_canvas_image(canvas_result.image_data)

#         # Предсказание с использованием предварительно обученной модели
#         result = model.predict(img_array)
#         predicted_class = np.argmax(result)

#         st.success(f'Предсказанная цифра: {predicted_class}')
#     except Exception as e:
#         st.error(f'Ошибка: {e}')
############################################################################################

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import matplotlib.pyplot as plt

# Задаем параметры холста
drawing_mode = "freedraw"
stroke_width = 12
stroke_color = "black"  # Устанавливаем черный цвет для контура
bg_color = "white"  # Устанавливаем белый цвет фона
bg_image = st.sidebar.file_uploader("Фоновое изображение:", type=["png", "jpg"])
realtime_update = st.sidebar.checkbox("Обновление в реальном времени", True)

# Создаем компонент квадратной канвы
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Цвет заливки с некоторой прозрачностью
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    background_image=Image.open(bg_image) if bg_image else None,
    update_streamlit=realtime_update,
    height=150,  # Установим высоту и ширину канвы одинаковыми (в пикселях)
    width=150,
    drawing_mode=drawing_mode,
    key="canvas",
)

# Отображаем нарисованное изображение
# if canvas_result.image_data is not None:
#     st.image(canvas_result.image_data)

# Загрузка предварительно обученной модели
model = tf.keras.models.load_model('mnist_cnn_model.h5')

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

# Streamlit App
st.title('MNIST Digit Classifier')

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




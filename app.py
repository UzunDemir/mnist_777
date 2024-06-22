import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

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
    # Преобразуем массив данных изображения в объект PIL Image
    drawn_image = drawn_image.fromarray(canvas_result.image_data.astype('uint8'))

    # Сохраняем изображение в файл (вы можете указать свой путь и формат)
    drawn_image.save("нарисованное_изображение.png")
    st.success("Изображение успешно сохранено!")  


# Load the pre-trained model
model = tf.keras.models.load_model('mnist_cnn_model.h5')

# Function to preprocess the uploaded image
def preprocess_image(image):
    img = Image.open(image)
    img = img.convert('L')  # Convert to grayscale
    img = img.resize((28, 28))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape((1, 28, 28, 1))
    return img_array
# Streamlit App
st.title('MNIST Digit Classifier')

uploaded_image = image_data #st.file_uploader("Upload a digit image (MNIST format)", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = uploaded_image #Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img, caption='Uploaded Image (Resized)', use_column_width=True)

    with col2:
        st.write("")
        if st.button('Classify', key='classify_btn'):
            try:
                # Preprocess the uploaded image
                img_array = preprocess_image(uploaded_image)

                # Make a prediction using the pre-trained model
                result = model.predict(img_array)
                predicted_class = np.argmax(result)

                st.success(f'Predicted Digit: {predicted_class}')
            except Exception as e:
                st.error(f'Error: {e}')

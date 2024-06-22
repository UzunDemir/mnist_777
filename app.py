import streamlit as st
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
    # Преобразуем массив данных изображения в объект PIL Image
    drawn_image = Image.fromarray(canvas_result.image_data.astype('uint8'))

    # Сохраняем изображение в файл (вы можете указать свой путь и формат)
    drawn_image.save("нарисованное_изображение.png")
    st.success("Изображение успешно сохранено!")

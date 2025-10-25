import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_drawable_canvas import st_canvas

# CONFIGURACIÓN DE LA PÁGINA
st.set_page_config(page_title='Reconocimiento de Dígitos escritos a mano', layout='centered')

# ESTILOS PERSONALIZADOS
st.markdown("""
    <style>
        body {
            background-color: #fcd4e4; /* Rosado pastel claro */
            font-family: 'Trebuchet MS', sans-serif;
            color: #2e2e2e;
        }
        .main {
            background-color: rgba(255, 255, 255, 0.9);
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0px 0px 25px rgba(0, 0, 0, 0.08);
        }
        h1, h2, h3, h4 {
            text-align: center;
            font-family: 'Comic Sans MS', cursive, sans-serif;
            color: #ad1457;
        }
        .stButton button {
            background-color: #fcb3cc;
            color: white;
            border-radius: 10px;
            font-size: 18px;
            transition: 0.3s;
        }
        .stButton button:hover {
            background-color: #ec407a;
            transform: scale(1.05);
        }
    </style>
""", unsafe_allow_html=True)

# TÍTULOS
st.title('💗 Reconocimiento de Dígitos escritos a mano')
st.subheader("✍️ Dibuja el dígito en el panel y presiona 'Predecir'")

# CONFIGURACIÓN DEL CANVAS
stroke_width = st.slider('Selecciona el ancho de línea', 1, 30, 15)
stroke_color = '#FFFFFF'
bg_color = '#000000'

canvas_result = st_canvas(
    fill_color="rgba(255, 182, 193, 0.3)",  # Rosado suave translúcido
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    height=400,  # aumentado
    width=400,   # aumentado
    key="canvas",
)

# FUNCIÓN DE PREDICCIÓN
def predictDigit(image):
    model = tf.keras.models.load_model("model/handwritten.h5")
    image = ImageOps.grayscale(image)
    img = image.resize((28, 28))
    img = np.array(img, dtype='float32') / 255
    plt.imshow(img)
    plt.show()
    img = img.reshape((1, 28, 28, 1))
    pred = model.predict(img)
    result = np.argmax(pred[0])
    return result

# BOTÓN DE PREDICCIÓN
if st.button('🔍 Predecir'):
    if canvas_result.image_data is not None:
        input_numpy_array = np.array(canvas_result.image_data)
        input_image = Image.fromarray(input_numpy_array.astype('uint8'), 'RGBA')
        input_image.save('prediction/img.png')
        img = Image.open("prediction/img.png")
        res = predictDigit(img)
        st.success(f'✨ El dígito es: **{res}**')
    else:
        st.warning('Por favor dibuja en el panel antes de predecir.')

# SIDEBAR
st.sidebar.title("ℹ️ Acerca de:")
st.sidebar.info("""
En esta aplicación se evalúa  
la capacidad de una RNA para reconocer  
dígitos escritos a mano.  

Basado en el desarrollo de Vinay Uniyal.
""")



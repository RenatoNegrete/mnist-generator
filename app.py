import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="MNIST Digit Generator", layout="centered")
st.title("Digit Generator - MNIST GAN")

@st.cache_resource
def load_generator():
    return tf.keras.models.load.model('generator_model')

generator = load_generator()

latent_dim = 100
num_classes = 10

digit = st.number_input("Enter a digit (0-9): ", min_value=0, max_value=9, step=1, value=0)

if st.button("Generate Images"):
    noise = tf.random.normal([5, latent_dim])
    labels = tf.one_hot([digit] * 5, num_classes)
    generated_images = generator([noise, labels], training=False)

    st.write(f"Generated images for the digit `{digit}`:")
    cols = st.columns(5)

    for i, col in enumerate(cols):
        img_arry = generated_images[i, :, :, 0].numpy() * 255
        img = Image.fromarray(img_arry.astype(np.uint8), mode='L')
        col.image(img.resize((112, 112)), caption=f"#{i+1}")
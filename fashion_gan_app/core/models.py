"""Model loading (cached once per server process) and model-file checks."""
from pathlib import Path

import numpy as np
import streamlit as st

APP_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = APP_DIR / "models"
ASSET_DIR = APP_DIR / "assets"

CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
CLASS_ICONS = ["👕", "👖", "🧥", "👗", "🧥", "🩴", "👔", "👟", "👜", "🥾"]
GENERATOR_FILE = "fashion_generator.keras"
DISCRIMINATOR_FILE = "fashion_discriminator.keras"
JUDGE_FILE = "fashion_judge.keras"
REAL_SAMPLES_FILE = "real_samples.npz"


def has(file):
    return (MODEL_DIR / file).exists()


def require_generator():
    """Stop the page with instructions if the trained generator is not in models/."""
    if not has(GENERATOR_FILE):
        st.error(
            f"Model file not found: `models/{GENERATOR_FILE}`.\n\n"
            "Run `fashion_conditional_gan.ipynb` on Colab, download `fashion_gan_results.zip`, and copy "
            "its `models/` files into `fashion_gan_app/models/`."
        )
        st.stop()


@st.cache_resource(show_spinner="Loading model…")
def load_model(file):
    import keras

    return keras.models.load_model(MODEL_DIR / file, compile=False)


@st.cache_resource
def load_real_samples():
    d = np.load(MODEL_DIR / REAL_SAMPLES_FILE)
    return d["x"], d["y"]


def latent_dim():
    return int(load_model(GENERATOR_FILE).inputs[0].shape[-1])


def to_model_input(imgs_u8):
    """uint8 images (n, 28, 28) → float32 (n, 28, 28, 1) in [-1, 1], the range the models were trained on."""
    return imgs_u8[..., None].astype("float32") / 127.5 - 1.0


def run_generator(noise, labels):
    """noise: (n, LATENT) float32, labels: (n,) int → uint8 images (n, 28, 28)."""
    import tensorflow as tf

    imgs = load_model(GENERATOR_FILE)([tf.constant(noise, tf.float32),
                                       tf.constant(np.asarray(labels)[:, None], tf.int32)], training=False).numpy()
    return ((imgs[..., 0] + 1) * 127.5).clip(0, 255).astype(np.uint8)


def judge_predict(imgs_u8):
    """Judge classifier's class probabilities."""
    return load_model(JUDGE_FILE).predict(to_model_input(imgs_u8), verbose=0)


def real_probability(imgs_u8, labels):
    """Discriminator's probability that each image is a real image of the given category."""
    logits = load_model(DISCRIMINATOR_FILE).predict(
        [to_model_input(imgs_u8), np.asarray(labels, np.int32)[:, None]], verbose=0).ravel()
    return 1 / (1 + np.exp(-logits))

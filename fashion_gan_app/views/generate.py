import io
import zipfile

import numpy as np
import streamlit as st
from PIL import Image

from core.models import (CLASS_ICONS, CLASS_NAMES, JUDGE_FILE, has, judge_predict, latent_dim, require_generator,
                         run_generator)

st.title("🎨 Generate fashion products")
st.caption("Conditional GAN (cGAN) trained on Fashion-MNIST")
st.info("Every image on this page is **synthetic** — generated from random noise, not photographed.")

require_generator()
LATENT = latent_dim()
LABELS = [f"{icon} {name}" for icon, name in zip(CLASS_ICONS, CLASS_NAMES)]


def noise(seed, n):
    return np.random.default_rng(seed).standard_normal((n, LATENT)).astype("float32")


@st.cache_data(max_entries=32)
def generate(seed: int, n: int, cls: int):
    return run_generator(noise(seed, n), np.full(n, cls))


@st.cache_data(max_entries=16)
def generate_all_classes(seed: int, n: int):
    z = np.tile(noise(seed, n), (len(CLASS_NAMES), 1))
    imgs = run_generator(z, np.repeat(np.arange(len(CLASS_NAMES)), n))
    return imgs.reshape(len(CLASS_NAMES), n, *imgs.shape[1:])


@st.cache_data(max_entries=16)
def interpolate(seed_a: int, seed_b: int, cls: int, steps: int):
    za, zb = noise(seed_a, 1)[0], noise(seed_b, 1)[0]
    z = np.stack([(1 - t) * za + t * zb for t in np.linspace(0, 1, steps)]).astype("float32")
    return run_generator(z, np.full(steps, cls))


def upscale(a, size=168):
    return Image.fromarray(a).resize((size, size), Image.NEAREST if size % a.shape[0] == 0 else Image.BICUBIC)


def show_row(imgs, cols, captions=None):
    for i in range(0, len(imgs), cols):
        row = st.columns(cols)
        for j, im in enumerate(imgs[i:i + cols]):
            row[j].image(upscale(im), caption=captions[i + j] if captions else None, width="stretch")


with st.sidebar:
    mode = st.radio("Mode", ["One category", "All categories (same noise)"])
    if mode == "One category":
        cls = st.selectbox("Category", range(len(CLASS_NAMES)), format_func=LABELS.__getitem__,
                           index=CLASS_NAMES.index("Sneaker"))
        n = st.slider("Number of images", 4, 32, 16, 4)
    else:
        n = st.slider("Designs per category", 2, 8, 6)
    st.session_state.setdefault("seed", 42)
    seed = st.number_input("Random seed", 0, 1_000_000, key="seed")
    st.button("🎲 New random seed", width="stretch",
              on_click=lambda: st.session_state.update(seed=int(np.random.default_rng().integers(1_000_000))))
    show_judge = st.toggle("Show judge's verdict", value=has(JUDGE_FILE), disabled=not has(JUDGE_FILE),
                           help="An independent CNN trained on real Fashion-MNIST names what it sees in each image.")

st.subheader("Generated images")
st.caption(f"Real size 28×28, shown enlarged. Seed {seed}.")

if mode == "One category":
    imgs = generate(int(seed), n, cls)
    captions = None
    if show_judge:
        p = judge_predict(imgs)
        captions = [("✅ " if k == cls else "❌ ") + f"{CLASS_NAMES[k]} {pk[k]:.0%}" for k, pk in zip(p.argmax(1), p)]
        hits = int((p.argmax(1) == cls).sum())
        st.metric("Judge recognises the requested category", f"{hits} / {n}")
    show_row(list(imgs), 8 if n >= 8 else n, captions)
    batch = [(f"{CLASS_NAMES[cls].replace('/', '_').replace(' ', '_').lower()}_{i:02d}.png", im)
             for i, im in enumerate(imgs)]
else:
    st.markdown("Each **column** uses the **same noise vector** — only the category label changes. "
                "Notice how brightness and texture carry over between categories.")
    grid = generate_all_classes(int(seed), n)
    for c, row_imgs in enumerate(grid):
        cols = st.columns([1.3] + [1] * n)
        cols[0].markdown(f"<div style='padding-top:1.4em'><b>{LABELS[c]}</b></div>", unsafe_allow_html=True)
        for j, im in enumerate(row_imgs):
            cols[j + 1].image(upscale(im, 112), width="stretch")
    batch = [(f"{CLASS_NAMES[c].replace('/', '_').replace(' ', '_').lower()}_{j:02d}.png", im)
             for c, row_imgs in enumerate(grid) for j, im in enumerate(row_imgs)]

buf = io.BytesIO()
with zipfile.ZipFile(buf, "w") as zf:
    for name, im in batch:
        png = io.BytesIO()
        Image.fromarray(im).save(png, format="PNG")
        zf.writestr(name, png.getvalue())
st.download_button(f"Download these {len(batch)} images (PNG, 28×28)", buf.getvalue(),
                   f"synthetic_fashion_seed{seed}.zip", "application/zip")

st.divider()
st.subheader("Latent-space interpolation")
st.markdown("Walking in a straight line between two noise vectors. Smooth transitions mean the GAN learned a "
            "continuous space of designs rather than memorising individual training images.")
c1, c2, c3, c4 = st.columns(4)
seed_a = c1.number_input("Seed A", 0, 1_000_000, 1)
seed_b = c2.number_input("Seed B", 0, 1_000_000, 2)
icls = c3.selectbox("Category ", range(len(CLASS_NAMES)), format_func=LABELS.__getitem__,
                    index=CLASS_NAMES.index("Dress"))
steps = c4.slider("Steps", 4, 12, 8)
show_row(list(interpolate(int(seed_a), int(seed_b), icls, steps)), steps,
         [f"{t:.2f}" for t in np.linspace(0, 1, steps)])

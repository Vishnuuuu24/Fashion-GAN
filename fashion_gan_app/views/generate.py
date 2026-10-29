import io
import time
import zipfile

import numpy as np
import streamlit as st
from PIL import Image

from core.models import (CLASS_ICONS, CLASS_NAMES, JUDGE_FILE, has, judge_predict, latent_dim, require_generator,
                         run_generator)

st.title("🎨 Generate")

require_generator()
LATENT = latent_dim()
LABELS = [f"{icon} {name}" for icon, name in zip(CLASS_ICONS, CLASS_NAMES)]


def noise(seed, n):
    return np.random.default_rng(seed).standard_normal((n, LATENT)).astype("float32")


def upscale(a, size=168):
    return Image.fromarray(a).resize((size, size), Image.NEAREST)


def show_grid(imgs, cols, captions=None, size=168):
    for i in range(0, len(imgs), cols):
        row = st.columns(cols)
        for j, im in enumerate(imgs[i:i + cols]):
            row[j].image(upscale(im, size), caption=captions[i + j] if captions else None, width="stretch")


def new_images():
    st.session_state.seed = int(np.random.default_rng().integers(1_000_000))
    st.session_state.fresh = True


st.session_state.setdefault("seed", 42)

mode = st.radio("Mode", ["One category", "All categories"], horizontal=True, label_visibility="collapsed")
c1, c2, c3 = st.columns([2, 2, 1], vertical_alignment="bottom")
if mode == "One category":
    cls = c1.selectbox("Category", range(len(CLASS_NAMES)), format_func=LABELS.__getitem__,
                       index=CLASS_NAMES.index("Sneaker"))
    n = c2.slider("Images", 4, 32, 16, 4)
else:
    n = c2.slider("Designs per category", 2, 8, 6)
c3.button("✨ Generate", type="primary", width="stretch", on_click=new_images)

with st.expander("Options"):
    seed = st.number_input("Seed (same seed = same images)", 0, 1_000_000, key="seed")
    show_judge = st.toggle("Show judge's verdict", value=has(JUDGE_FILE), disabled=not has(JUDGE_FILE))

start = time.perf_counter()
if mode == "One category":
    imgs = run_generator(noise(int(seed), n), np.full(n, cls))
    batch = list(imgs)
else:
    z = np.tile(noise(int(seed), n), (len(CLASS_NAMES), 1))
    grid = run_generator(z, np.repeat(np.arange(len(CLASS_NAMES)), n)).reshape(len(CLASS_NAMES), n, 28, 28)
    batch = list(grid.reshape(-1, 28, 28))
elapsed = time.perf_counter() - start

if st.session_state.pop("fresh", False):
    st.success(f"Generated {len(batch)} new images in {elapsed:.2f} s (seed {seed})")

if mode == "One category":
    captions = None
    if show_judge:
        p = judge_predict(imgs)
        pred = p.argmax(1)
        captions = [("✅ " if k == cls else "❌ ") + f"{CLASS_NAMES[k]} {pk[k]:.0%}" for k, pk in zip(pred, p)]
        st.caption(f"Judge agrees on {int((pred == cls).sum())} of {n}")
    show_grid(imgs, 8, captions)
    names = [f"{CLASS_NAMES[cls].replace('/', '_').lower()}_{i:02d}.png" for i in range(n)]
else:
    st.caption("Each column uses the same noise. Only the label changes.")
    for c, row_imgs in enumerate(grid):
        cols = st.columns([1.4] + [1] * n, vertical_alignment="center")
        cols[0].markdown(f"**{LABELS[c]}**")
        for j, im in enumerate(row_imgs):
            cols[j + 1].image(upscale(im, 112), width="stretch")
    names = [f"{CLASS_NAMES[c].replace('/', '_').lower()}_{j:02d}.png" for c in range(len(CLASS_NAMES)) for j in range(n)]

buf = io.BytesIO()
with zipfile.ZipFile(buf, "w") as zf:
    for name, im in zip(names, batch):
        png = io.BytesIO()
        Image.fromarray(im).save(png, format="PNG")
        zf.writestr(name, png.getvalue())
st.download_button(f"⬇️ Download {len(batch)} images", buf.getvalue(), f"fashion_gan_seed{seed}.zip",
                   "application/zip")

st.divider()
st.subheader("Morph between two designs")
c1, c2, c3 = st.columns(3)
seed_a = c1.number_input("Seed A", 0, 1_000_000, 1)
seed_b = c2.number_input("Seed B", 0, 1_000_000, 2)
icls = c3.selectbox("Category ", range(len(CLASS_NAMES)), format_func=LABELS.__getitem__,
                    index=CLASS_NAMES.index("Dress"))
za, zb = noise(int(seed_a), 1)[0], noise(int(seed_b), 1)[0]
steps = np.linspace(0, 1, 8)
morph = run_generator(np.stack([(1 - t) * za + t * zb for t in steps]).astype("float32"), np.full(8, icls))
show_grid(morph, 8)

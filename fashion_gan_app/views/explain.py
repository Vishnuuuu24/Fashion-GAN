import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from core.models import (CLASS_ICONS, CLASS_NAMES, DISCRIMINATOR_FILE, REAL_SAMPLES_FILE, has, latent_dim,
                         load_real_samples, real_probability, require_generator, run_generator)

st.title("🔍 Input → Output")

require_generator()
LATENT = latent_dim()
SIDE = int(np.sqrt(LATENT))
LABELS = [f"{icon} {name}" for icon, name in zip(CLASS_ICONS, CLASS_NAMES)]


def heatmap(values, side, cell=18, vmax=2.5):
    """Noise vector as a grid: red > 0, blue < 0."""
    v = np.clip(np.asarray(values).reshape(side, side) / vmax, -1, 1)
    rgb = np.ones((side, side, 3))
    rgb[..., 0] -= np.clip(-v, 0, 1)
    rgb[..., 1] -= np.abs(v)
    rgb[..., 2] -= np.clip(v, 0, 1)
    return Image.fromarray((rgb * 255).astype(np.uint8)).resize((side * cell, side * cell), Image.NEAREST)


def diff_map(a, b, size=168):
    """|a - b|: black = same, yellow = very different."""
    d = np.abs(a.astype(float) - b.astype(float)) / 255
    rgb = np.stack([np.clip(d * 3, 0, 1), np.clip(d * 3 - 1, 0, 1), np.clip(d * 3 - 2, 0, 1)], -1)
    return Image.fromarray((rgb * 255).astype(np.uint8)).resize((size, size), Image.NEAREST)


def big(a, size=168):
    return Image.fromarray(a).resize((size, size), Image.NEAREST)


st.session_state.setdefault("xseed", 7)
c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
cls = c1.selectbox("Category", range(len(CLASS_NAMES)), format_func=LABELS.__getitem__,
                   index=CLASS_NAMES.index("Sneaker"))
c2.button("🎲 New noise", type="primary", width="stretch",
          on_click=lambda: st.session_state.update(xseed=int(np.random.default_rng().integers(1_000_000))))

noise = np.random.default_rng(st.session_state.xseed).standard_normal((1, LATENT)).astype("float32")
out = run_generator(noise, [cls])[0]

st.subheader("1. Inputs and output")
c1, c2, c3, c4 = st.columns([1.3, 1.3, 0.3, 1])
with c1:
    st.markdown("**Noise** (100 numbers)")
    st.image(heatmap(noise[0], SIDE), width="stretch")
    st.caption("Red > 0, blue < 0. Controls the style.")
with c2:
    st.markdown("**Label** (one-hot)")
    st.dataframe(pd.DataFrame({"category": LABELS, "value": np.eye(len(CLASS_NAMES), dtype=int)[cls]}),
                 hide_index=True, height=388)
with c3:
    st.markdown("<div style='font-size:3em;padding-top:2.5em;text-align:center'>→</div>", unsafe_allow_html=True)
with c4:
    st.markdown("**Output** (28×28)")
    st.image(big(out), width="stretch")

st.subheader("2. What each input controls")
st.caption("Same noise, every label: the label sets **what** is drawn.")
cols = st.columns(len(CLASS_NAMES))
for c, im in enumerate(run_generator(np.repeat(noise, len(CLASS_NAMES), 0), np.arange(len(CLASS_NAMES)))):
    cols[c].image(big(im, 112), caption=CLASS_NAMES[c], width="stretch")
st.caption(f"Same label, new noise: the noise sets the **style**.")
other = np.random.default_rng(st.session_state.xseed + 1).standard_normal((10, LATENT)).astype("float32")
cols = st.columns(10)
for j, im in enumerate(run_generator(other, np.full(10, cls))):
    cols[j].image(big(im, 112), width="stretch")

if has(REAL_SAMPLES_FILE):
    x_r, y_r = load_real_samples()
    pool = x_r[y_r == cls]

    st.subheader("3. Output vs closest real image")
    dists = np.sqrt(((pool.astype(float) - out.astype(float)) ** 2).sum((1, 2))) / 127.5
    near = pool[dists.argmin()]
    c1, c2, c3 = st.columns(3)
    c1.image(big(out), caption="Generated", width="stretch")
    c2.image(big(near), caption="Closest real", width="stretch")
    c3.image(diff_map(out, near), caption=f"Difference (distance {dists.min():.1f})", width="stretch")
    st.caption("A clear difference means the GAN made a new image, not a copy.")

    st.subheader("4. Average real vs average generated")
    gen_avg = run_generator(np.random.default_rng(0).standard_normal((100, LATENT)).astype("float32"),
                            np.full(100, cls)).mean(0)
    real_avg = pool.mean(0)
    c1, c2, c3 = st.columns(3)
    c1.image(big(real_avg.round().astype(np.uint8)), caption="Real average", width="stretch")
    c2.image(big(gen_avg.round().astype(np.uint8)), caption="Generated average", width="stretch")
    c3.image(diff_map(real_avg, gen_avg), caption="Difference", width="stretch")

    if has(DISCRIMINATOR_FILE):
        st.subheader("5. Discriminator's verdict")
        wrong = (cls + 1) % len(CLASS_NAMES)
        c1, c2, c3 = st.columns(3)
        c1.metric("This output", f"{real_probability(out[None], [cls])[0]:.0%} real")
        c2.metric(f"Real {CLASS_NAMES[cls]} (avg)", f"{real_probability(pool[:100], np.full(100, cls)).mean():.0%} real")
        c3.metric(f"Output labelled {CLASS_NAMES[wrong]}", f"{real_probability(out[None], [wrong])[0]:.0%} real")
        st.caption("Near 50% for both real and generated means the generator fools the discriminator. "
                   "A wrong label lowers the score.")

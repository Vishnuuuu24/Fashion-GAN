import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from core.models import (CLASS_ICONS, CLASS_NAMES, DISCRIMINATOR_FILE, REAL_SAMPLES_FILE, has, latent_dim,
                         load_real_samples, real_probability, require_generator, run_generator)

st.title("🔍 Input → Output")
st.markdown(
    "What goes **into** the generator, what comes **out**, and how the output differs from real images. "
    "Change the inputs in the sidebar and watch the output change."
)

require_generator()
LATENT = latent_dim()
SIDE = int(np.sqrt(LATENT))            # 100 noise numbers → 10 × 10 grid
LABELS = [f"{icon} {name}" for icon, name in zip(CLASS_ICONS, CLASS_NAMES)]

with st.sidebar:
    cls = st.selectbox("Category (input 2)", range(len(CLASS_NAMES)), format_func=LABELS.__getitem__,
                       index=CLASS_NAMES.index("Sneaker"))
    st.session_state.setdefault("xseed", 7)
    seed = st.number_input("Noise seed (input 1)", 0, 1_000_000, key="xseed")
    st.button("🎲 New noise", width="stretch",
              on_click=lambda: st.session_state.update(xseed=int(np.random.default_rng().integers(1_000_000))))


def heatmap(values, side, cell=18, vmax=2.5):
    """Blue (negative) → white (0) → red (positive) picture of the noise vector."""
    v = np.clip(np.asarray(values).reshape(side, side) / vmax, -1, 1)
    rgb = np.ones((side, side, 3))
    rgb[..., 0] -= np.clip(-v, 0, 1)
    rgb[..., 1] -= np.abs(v)
    rgb[..., 2] -= np.clip(v, 0, 1)
    return Image.fromarray((rgb * 255).astype(np.uint8)).resize((side * cell, side * cell), Image.NEAREST)


def diff_map(a, b, size=168):
    """|a − b| shown black (same) → red → yellow (very different)."""
    d = np.abs(a.astype(float) - b.astype(float)) / 255
    rgb = np.stack([np.clip(d * 3, 0, 1), np.clip(d * 3 - 1, 0, 1), np.clip(d * 3 - 2, 0, 1)], -1)
    return Image.fromarray((rgb * 255).astype(np.uint8)).resize((size, size), Image.NEAREST)


def big(a, size=168):
    return Image.fromarray(a).resize((size, size), Image.NEAREST)


noise = np.random.default_rng(int(seed)).standard_normal((1, LATENT)).astype("float32")
out = run_generator(noise, [cls])[0]

st.header("1. One generation, step by step")
c1, c2, c3, c4 = st.columns([1.3, 1.3, 0.3, 1])
with c1:
    st.markdown("**Input 1 — noise z**")
    st.image(heatmap(noise[0], SIDE), width="stretch")
    st.caption(f"{LATENT} random numbers (normal distribution), drawn as a {SIDE} × {SIDE} grid: red > 0, blue < 0. "
               "They decide the **style**.")
with c2:
    st.markdown("**Input 2 — category label**")
    st.dataframe(pd.DataFrame({"category": LABELS, "one-hot": np.eye(len(CLASS_NAMES), dtype=int)[cls]}),
                 hide_index=True, height=388)
with c3:
    st.markdown("<div style='font-size:3em;padding-top:2.5em;text-align:center'>→</div>", unsafe_allow_html=True)
with c4:
    st.markdown("**Output — generated image**")
    st.image(big(out), width="stretch")
    st.caption(f"28 × 28 grayscale image. Requested: {LABELS[cls]}")

st.header("2. Which input controls what?")
st.markdown("**Same noise, every label** → the label decides **what** is drawn.")
all_cls = run_generator(np.repeat(noise, len(CLASS_NAMES), 0), np.arange(len(CLASS_NAMES)))
cols = st.columns(len(CLASS_NAMES))
for c, im in enumerate(all_cls):
    cols[c].image(big(im, 112), caption=CLASS_NAMES[c], width="stretch")
st.markdown(f"**Same label ({CLASS_NAMES[cls]}), different noise** → the noise decides the **style**.")
other = np.random.default_rng(int(seed) + 1).standard_normal((10, LATENT)).astype("float32")
cols = st.columns(10)
for j, im in enumerate(run_generator(other, np.full(10, cls))):
    cols[j].image(big(im, 112), width="stretch")

if has(REAL_SAMPLES_FILE):
    x_r, y_r = load_real_samples()
    pool = x_r[y_r == cls]

    st.header("3. Output vs the closest real image")
    st.markdown(
        f"We search {len(pool)} real {CLASS_NAMES[cls]} images for the one most similar to the output and show where "
        "they differ. If the GAN were copying training images, the difference would be almost black."
    )
    dists = np.sqrt(((pool.astype(float) - out.astype(float)) ** 2).sum((1, 2))) / 127.5
    near = pool[dists.argmin()]
    c1, c2, c3 = st.columns(3)
    c1.image(big(out), caption="generated (output)", width="stretch")
    c2.image(big(near), caption="closest real image", width="stretch")
    c3.image(diff_map(out, near), caption=f"difference (distance {dists.min():.1f})", width="stretch")

    st.header("4. Average real image vs average generated image")
    gen_avg = run_generator(np.random.default_rng(0).standard_normal((100, LATENT)).astype("float32"),
                            np.full(100, cls)).mean(0)
    real_avg = pool.mean(0)
    c1, c2, c3 = st.columns(3)
    c1.image(big(real_avg.round().astype(np.uint8)), caption=f"average of {len(pool)} real images", width="stretch")
    c2.image(big(gen_avg.round().astype(np.uint8)), caption="average of 100 generated images", width="stretch")
    c3.image(diff_map(real_avg, gen_avg), caption="difference", width="stretch")

    if has(DISCRIMINATOR_FILE):
        st.header("5. What does the discriminator think?")
        st.markdown("The trained discriminator gives the probability that an image is a **real** image of the given "
                    "category. It was the generator's only teacher during training.")
        wrong = (cls + 1) % len(CLASS_NAMES)
        rand = np.random.default_rng(1).integers(0, 256, (1, 28, 28)).astype(np.uint8)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("this output", f"{real_probability(out[None], [cls])[0]:.0%}")
        c2.metric(f"real {CLASS_NAMES[cls]} images (average)",
                  f"{real_probability(pool[:100], np.full(100, cls)).mean():.0%}")
        c3.metric(f"this output, but labelled {CLASS_NAMES[wrong]}", f"{real_probability(out[None], [wrong])[0]:.0%}")
        c4.metric("random noise", f"{real_probability(rand, [cls])[0]:.0%}")
        st.caption("A well-trained discriminator ends up unsure (≈ 50 %) about both real and generated images — "
                   "that means the generator is good at fooling it.")

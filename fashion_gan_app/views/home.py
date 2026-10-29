import streamlit as st

from core.models import (CLASS_ICONS, CLASS_NAMES, DISCRIMINATOR_FILE, GENERATOR_FILE, JUDGE_FILE, ASSET_DIR,
                         has)

st.title("👗 Fashion GAN Studio")
st.markdown(
    "A **Conditional GAN** that draws new clothing images of the category you choose — trained from scratch on "
    "**Fashion-MNIST** (70,000 images, 10 categories)."
)
st.markdown(" ".join(f"{icon} {name}" for icon, name in zip(CLASS_ICONS, CLASS_NAMES)))

c1, c2, c3 = st.columns(3)
with c1, st.container(border=True):
    st.subheader("🎨 Generate")
    st.markdown("Pick a category and generate new images, see all categories with the same noise, and morph "
                "between two designs.")
    st.page_link("views/generate.py", label="Open the generator", icon="➡️")
with c2, st.container(border=True):
    st.subheader("🔍 Input → Output")
    st.markdown("See what goes in (noise + label), what comes out, how it differs from real images, and what the "
                "discriminator thinks.")
    st.page_link("views/explain.py", label="Open input → output", icon="➡️")
with c3, st.container(border=True):
    st.subheader("📊 Results")
    st.markdown("Three discriminators compared (Vanilla, cGAN, AC-GAN), training progress, and the log of every "
                "training run.")
    st.page_link("views/results.py", label="Open the results", icon="➡️")

if (ASSET_DIR / "training.gif").exists():
    st.image(str(ASSET_DIR / "training.gif"), caption="Training progress — rows are categories", width=360)

st.divider()
st.subheader("How it works")
st.markdown(
    """
* **Generator** — takes **100 random numbers** (noise) + a **category label** and turns them into a 28 × 28 image
  using Dense and Conv2DTranspose layers.
* **Discriminator** — a CNN that looks at an image + its label and outputs the probability that it is **real**.
* **Training** — the two networks take turns: D learns to spot fakes, G learns to fool D. Both use binary
  cross-entropy loss and the Adam optimizer.
"""
)

st.subheader("Model status")
for name, f in [("Generator", GENERATOR_FILE), ("Discriminator", DISCRIMINATOR_FILE), ("Judge classifier", JUDGE_FILE)]:
    st.markdown(f"✅ **{name}** — `models/{f}`" if has(f) else
                f"❌ **{name}** — missing `models/{f}` (train with `fashion_conditional_gan.ipynb`)")

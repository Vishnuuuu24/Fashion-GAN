import pandas as pd
import streamlit as st

from core.models import ASSET_DIR, DISCRIMINATOR_FILE, GENERATOR_FILE, JUDGE_FILE, has

st.title("👗 Fashion GAN")
st.markdown("A conditional GAN that draws clothing of the category you choose. Built from scratch, "
            "trained on Fashion-MNIST.")

log = ASSET_DIR / "training_log.csv"
if log.exists():
    final = pd.read_csv(log).iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Class accuracy", f"{final['class accuracy']:.1%}")
    c2.metric("Categories", "10")
    c3.metric("Image size", "28 × 28")
    c4.metric("Training", f"{int(final['epochs'])} epochs")

c1, c2, c3 = st.columns(3)
with c1, st.container(border=True):
    st.markdown("**🎨 Generate**  \nCreate new images of any category.")
    st.page_link("views/generate.py", label="Open", icon="➡️")
with c2, st.container(border=True):
    st.markdown("**🔍 Input → Output**  \nWhat goes in, what comes out.")
    st.page_link("views/explain.py", label="Open", icon="➡️")
with c3, st.container(border=True):
    st.markdown("**📊 Results**  \nDiscriminators compared, training log.")
    st.page_link("views/results.py", label="Open", icon="➡️")

left, right = st.columns([1, 1.4], gap="large")
with left:
    if (ASSET_DIR / "training.gif").exists():
        st.image(str(ASSET_DIR / "training.gif"), caption="Training, epoch by epoch")
with right:
    st.subheader("How it works")
    st.markdown(
        """
| | Generator | Discriminator |
|---|---|---|
| Input | 100 random numbers + category | image + category |
| Output | 28×28 image | probability it is real |
| Goal | fool the discriminator | catch fakes |

They train in turns with binary cross-entropy and Adam.
"""
    )

missing = [f for f in (GENERATOR_FILE, DISCRIMINATOR_FILE, JUDGE_FILE) if not has(f)]
if missing:
    st.error("Missing model files in `models/`: " + ", ".join(f"`{f}`" for f in missing))

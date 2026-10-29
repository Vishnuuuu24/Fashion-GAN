import pandas as pd
import streamlit as st

from core.models import ASSET_DIR

st.title("📊 Results")

if not (ASSET_DIR / "training_log.csv").exists():
    st.warning("No results in `assets/`. Copy the notebook's `gan_outputs/` files there.")
    st.stop()


def image(name, caption=None, width="stretch"):
    if (ASSET_DIR / name).exists():
        st.image(str(ASSET_DIR / name), caption=caption, width=width)


def table(name):
    return pd.read_csv(ASSET_DIR / name, index_col=0)


tab1, tab2, tab3, tab4 = st.tabs(["Discriminators", "Training", "Real vs generated", "Training log"])

with tab1:
    st.markdown(
        """
| Discriminator | Label use | Output |
|---|---|---|
| Vanilla GAN | none | real/fake |
| cGAN | label added as an image channel | real/fake |
| AC-GAN | predicts the label | real/fake + class |
"""
    )
    st.caption("Same generator and 10 epochs each. Class accuracy = % of images a separate classifier labels as "
               "the requested category (random = 10%).")
    cmp = table("discriminator_comparison.csv")
    c1, c2 = st.columns([1, 1])
    c1.dataframe(cmp.style.format({"class accuracy": "{:.1%}", "final D loss": "{:.3f}", "final G loss": "{:.3f}",
                                   "training time (s)": "{:.0f}"}), width="stretch")
    with c2:
        image("discriminator_comparison.png")
    st.markdown("**Samples** (rows = requested category)")
    cols = st.columns(3)
    for col, kind, name in zip(cols, ["vanilla", "cgan", "acgan"], ["Vanilla", "cGAN", "AC-GAN"]):
        with col:
            st.markdown(f"**{name}**")
            image(f"cmp_{kind}.png")

with tab2:
    c1, c2 = st.columns(2)
    with c1:
        image("training.gif", "Final cGAN, one frame per epoch")
    with c2:
        image("losses.png", "Losses stay level: neither network wins")
        image("discriminator_scores.png", "Discriminator's probability 'real'")

with tab3:
    image("input_output.png", "Inputs and output")
    image("label_vs_noise.png", "Label sets what is drawn, noise sets the style")
    image("mean_difference.png", "Average images and their difference")
    image("nearest_difference.png", "Generated vs closest real image: new images, not copies")
    c1, c2 = st.columns([1, 1.1])
    c1.dataframe(table("fidelity.csv").style.format("{:.1%}"), width="stretch", height=420)
    with c2:
        image("confusion_matrix.png")

with tab4:
    st.caption("Every run is saved in `runs/`, including weaker ones.")
    log = pd.read_csv(ASSET_DIR / "training_log.csv")
    st.dataframe(log.style.format({"class accuracy": "{:.1%}", "learning rate": "{:g}"}), hide_index=True,
                 width="stretch")

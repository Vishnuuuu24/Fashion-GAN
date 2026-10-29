import pandas as pd
import streamlit as st

from core.models import ASSET_DIR

st.title("📊 Training & results")
st.caption("Produced by `fashion_conditional_gan.ipynb` on Google Colab (T4 GPU) and copied into `assets/`.")

if not (ASSET_DIR / "training_log.csv").exists():
    st.warning("No results found in `assets/`. Copy the notebook's `gan_outputs/` files there.")
    st.stop()


def image(name, caption=None, width="stretch"):
    if (ASSET_DIR / name).exists():
        st.image(str(ASSET_DIR / name), caption=caption, width=width)


def table(name):
    return pd.read_csv(ASSET_DIR / name, index_col=0)


st.header("1. Three discriminators compared")
st.markdown(
    """
Same generator, same data, same 10 epochs — only the **discriminator** changes.

| Discriminator | Uses the label? | Output |
|---|---|---|
| **Vanilla GAN** | ❌ no | real / fake |
| **cGAN** | ✅ label added to the image as an extra channel | real / fake |
| **AC-GAN** | ✅ must *predict* the label | real / fake **+** category |

**Class accuracy** = how often a separate classifier (the *judge*) recognises the category we asked for. Random guessing = 10 %.
"""
)
cmp = table("discriminator_comparison.csv")
st.dataframe(cmp.style.format({"class accuracy": "{:.1%}", "final D loss": "{:.3f}", "final G loss": "{:.3f}",
                               "training time (s)": "{:.0f}"}), width="stretch")
image("discriminator_comparison.png", width=560)
tabs = st.tabs(["Vanilla", "cGAN", "AC-GAN"])
for tab, kind in zip(tabs, ["vanilla", "cgan", "acgan"]):
    with tab:
        image(f"cmp_{kind}.png", "Each row = requested category, each column = the same noise", width=420)

st.header("2. Training the final cGAN")
c1, c2 = st.columns(2)
with c1:
    image("training.gif", "One picture per epoch", width=420)
with c2:
    image("losses.png", "Generator and discriminator losses")
    st.markdown("GAN losses don't go to zero: G and D compete, so the losses stay roughly level. "
                "If D's loss fell to ~0, D would be winning and G would stop learning.")

st.header("3. Input vs output")
image("input_output.png", "Inputs (noise + label) and the generated output")
image("label_vs_noise.png", "The label decides what is drawn, the noise decides the style")
image("mean_difference.png", "Average real image, average generated image and their difference")
image("nearest_difference.png", "Generated image, closest real training image and their difference — not copies")
image("discriminator_scores.png", "The discriminator's probability of 'real' for real, generated and noise images",
      width=620)

st.header("4. Does the judge recognise each category?")
c1, c2 = st.columns([1, 1.1])
with c1:
    st.dataframe(table("fidelity.csv").style.format("{:.1%}"), width="stretch", height=420)
with c2:
    image("confusion_matrix.png")

st.header("5. Training log — every run")
st.markdown("Every training run is saved (models in `models/runs/`) and logged, including runs that were not used.")
st.dataframe(pd.read_csv(ASSET_DIR / "training_log.csv"), hide_index=True, width="stretch")

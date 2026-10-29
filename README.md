# 👗 Fashion GAN — generating clothing images with a Conditional GAN

Chapter 9 project: **image generation with a GAN**, built from scratch in TensorFlow / Keras.

The GAN learns from **Fashion-MNIST** (70,000 grayscale 28 × 28 clothing images, 10 categories) and draws **new**
images of whatever category you ask for — T-shirt, trouser, sneaker, bag, …

| File | What it is |
|---|---|
| [`fashion_conditional_gan.ipynb`](fashion_conditional_gan.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Vishnuuuu24/Fashion-GAN/blob/main/fashion_conditional_gan.ipynb) | The notebook: theory, generator, **3 discriminators compared** (Vanilla GAN, cGAN, AC-GAN), training, input-vs-output analysis |
| [`fashion_gan_app/`](fashion_gan_app) | Streamlit app to generate images and explore input → output |
| [`TRAINING_LOG.md`](TRAINING_LOG.md) | Notes on every training run (settings, results, what changed) |
| [`STUDY_GUIDE.md`](STUDY_GUIDE.md) | Explanation of every part + likely viva questions and answers |

**Live app:** APP_URL

## How it works (short)
* **Generator** — input: 100 random numbers (noise) + a category label → output: a 28 × 28 image.
* **Discriminator** — input: an image + its label → output: probability that the image is real.
* They are trained in turns with **binary cross-entropy** and **Adam**: D learns to spot fakes, G learns to fool D.

## Train
Open the notebook in Colab → *Runtime → Change runtime type → T4 GPU* → *Run all* (≈ 10–15 min).
It downloads `fashion_gan_results.zip` at the end (all trained models, including every run in `models/runs/`, plus plots).

## Run the app
Copy the zip's `models/*` into `fashion_gan_app/models/` and `gan_outputs/*` into `fashion_gan_app/assets/`, then
```bash
cd fashion_gan_app
pip install -r requirements.txt
streamlit run app.py
```

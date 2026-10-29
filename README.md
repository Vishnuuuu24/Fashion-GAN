# 👗 Fashion GAN

Chapter 9 project: a conditional GAN, built from scratch, that generates clothing images of a chosen category.

**Live app:** https://fashion-gan-studio.streamlit.app

## Results
| Discriminator | Epochs | Class accuracy* |
|---|---|---|
| Vanilla GAN | 10 | 10.1% |
| cGAN | 10 | 81.6% |
| AC-GAN | 10 | 94.9% |
| **Final cGAN** | **30** | **89.3%** |

\* % of generated images a separate classifier labels as the requested category. Random = 10%. The classifier scores 90.3% on real images.

## Files
| Path | Contents |
|---|---|
| [`fashion_conditional_gan.ipynb`](fashion_conditional_gan.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Vishnuuuu24/Fashion-GAN/blob/main/fashion_conditional_gan.ipynb) | Notebook with all outputs |
| [`fashion_gan_app/`](fashion_gan_app) | Streamlit app |
| [`runs/`](runs) | Models from every training run |
| [`results/epochs/`](results/epochs) | Final run, one sample grid per epoch |
| [`TRAINING_LOG.md`](TRAINING_LOG.md) | Settings and results of each run |
| [`STUDY_GUIDE.md`](STUDY_GUIDE.md) | Explanations and viva Q&A |

## Data
Fashion-MNIST: 70,000 grayscale 28×28 images, 10 categories.

## Train
Colab → *Runtime → Change runtime type → T4 GPU* → *Run all* (about 25 min). Downloads `fashion_gan_results.zip` at the end.

## Run the app locally
Copy the zip's `models/*` into `fashion_gan_app/models/` and `gan_outputs/*` into `fashion_gan_app/assets/`, then:
```bash
cd fashion_gan_app
pip install -r requirements.txt
streamlit run app.py
```

# Training log

Every training attempt is recorded here. The models of every completed run are kept in `runs/<run name>/`
(generator + discriminator), even when the run was not the best, and the notebook appends each run to
`fashion_gan_app/assets/training_log.csv`.

Hardware: Google Colab, NVIDIA T4 GPU. Data: Fashion-MNIST, 60,000 training images.

---

## Attempt 0 — advanced version (stopped, no GAN trained) — 2026-10-29
* **Design:** 32 × 32 images, projection discriminator with spectral normalisation, hinge loss, DiffAugment, EMA
  generator, 4 discriminators, FID metric.
* **What happened:** stopped while it was still training the judge classifier, before any GAN training started, so
  there are no models to keep.
* **Why changed:** too advanced for the course; replaced with the standard DCGAN-style conditional GAN taught in the
  chapter (28 × 28, BCE loss, Adam, 3 discriminators: Vanilla / cGAN / AC-GAN).

---

## Attempt 1 — simple version (this notebook) — 2026-10-29
Settings for all runs: batch size 128, Adam (learning rate 0.0002, beta_1 0.5), latent dim 100, seed 42.
The judge classifier (used to measure "class accuracy") scores **90.3 %** on real test images — that is the ceiling.

| Run | Discriminator | Epochs | Final D loss | Final G loss | Class accuracy | Time | Models |
|---|---|---|---|---|---|---|---|
| `compare_vanilla_10ep` | Vanilla | 10 | 1.357 | 0.743 | **10.1 %** | 233 s | `runs/compare_vanilla_10ep/` |
| `compare_cgan_10ep` | cGAN | 10 | 1.288 | 0.817 | **81.6 %** | 239 s | `runs/compare_cgan_10ep/` |
| `compare_acgan_10ep` | AC-GAN | 10 | 1.672 | 0.794 | **94.9 %** | 237 s | `runs/compare_acgan_10ep/` |
| `final_cgan_30ep` | cGAN | 30 | 1.330 | 0.796 | **89.3 %** | 758 s | `runs/final_cgan_30ep/` (= the app's model) |

**Notes**
* **Vanilla (10.1 %)** — exactly chance level. Its discriminator never sees the label, so the generator ignored it:
  every row in `fashion_gan_app/assets/cmp_vanilla.png` shows the same items whatever category was requested. Kept as proof of
  *why* a conditional discriminator is needed — not used further.
* **cGAN, 10 → 30 epochs (81.6 % → 89.3 %)** — the only change between these two runs is the number of epochs.
  Training 3× longer improved class accuracy by ~8 points and the images became sharper.
* **AC-GAN (94.9 %)** — highest class accuracy, because its generator is directly rewarded for being classifiable.
  Not chosen as the final model: AC-GAN is known to give less variety inside a class, and cGAN is the standard
  conditional GAN taught in the chapter. (Possible next run: AC-GAN for 30 epochs, for a fair comparison.)
* **Final cGAN per category** — best: Trouser 99.4 %, Sandal 98.2 %, Bag 97.6 %; worst: Shirt 43.6 %, which the judge
  also gets only 60.9 % right on *real* shirts (shirts look like T-shirts, pullovers and coats).
* **Discriminator at the end** — average probability "real": real images 0.45, generated 0.39. D is unsure about both,
  so the GAN is close to balance. (Random noise got 0.37 — D never trained on noise, so that number is not meaningful.)
* Loss curves stayed level (D ≈ 1.3, G ≈ 0.8) for all runs — no collapse, no runaway discriminator.

Per-epoch sample grids of the final run: `results/epochs/epoch_01.png` … `epoch_30.png` (also `training.gif`).
Full machine-written log: `fashion_gan_app/assets/training_log.csv`.

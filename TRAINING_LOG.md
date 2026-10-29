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
Settings for all runs: batch size 128, Adam lr 0.0002, beta_1 0.5, latent dim 100, seed 42.

RUN1_TABLE

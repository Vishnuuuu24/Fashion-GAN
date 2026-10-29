# Training log

Every run's generator and discriminator are saved in `runs/<run name>/`.
Hardware: Colab T4 GPU. Data: Fashion-MNIST, 60,000 training images.

## Attempt 0: advanced version (stopped)
* **Design:** 32×32, projection discriminator, spectral norm, hinge loss, DiffAugment, EMA, FID.
* **Result:** stopped during judge training, before any GAN trained. No models.
* **Change:** too advanced for the course. Replaced with the standard DCGAN-style conditional GAN.

## Attempt 1: final version
**Settings (all runs):** batch 128, Adam (lr 0.0002, beta_1 0.5), latent dim 100, seed 42.
**Judge accuracy on real images:** 90.3% (the ceiling for class accuracy).

| Run | Discriminator | Epochs | D loss | G loss | Class accuracy | Time |
|---|---|---|---|---|---|---|
| `compare_vanilla_10ep` | Vanilla | 10 | 1.357 | 0.743 | 10.1% | 233 s |
| `compare_cgan_10ep` | cGAN | 10 | 1.288 | 0.817 | 81.6% | 239 s |
| `compare_acgan_10ep` | AC-GAN | 10 | 1.672 | 0.794 | 94.9% | 237 s |
| `final_cgan_30ep` | cGAN | 30 | 1.330 | 0.796 | **89.3%** | 758 s |

## Notes
* **Vanilla:** chance level. D ignores the label, so G does too. Every row in `cmp_vanilla.png` looks the same.
* **cGAN 10 → 30 epochs:** only the epoch count changed. Accuracy rose 81.6% → 89.3%, images got sharper.
* **AC-GAN:** highest accuracy, but less variety within a class. Not used as final model.
* **Per category (final):** best Trouser 99.4%, Sandal 98.2%, Bag 97.6%. Worst Shirt 43.6% (judge gets real shirts only 60.9%).
* **Discriminator at the end:** real 0.45, generated 0.39. Near balance. Noise 0.37 is not meaningful (D never saw noise).
* **Losses:** level in all runs (D ≈ 1.3, G ≈ 0.8). No collapse.

Full CSV: `fashion_gan_app/assets/training_log.csv`. Epoch grids: `results/epochs/`.

Trained model files from Colab (`fashion_gan_results.zip` → `models/`):

| File | Used for |
|---|---|
| `fashion_generator.keras` | generating images |
| `fashion_discriminator.keras` | "what does the discriminator think?" on the Input → Output page |
| `fashion_judge.keras` | the judge's verdict on each generated image |
| `real_samples.npz` | closest-real-image comparison |

Every training run's models are kept in the repository's `runs/` folder (see `TRAINING_LOG.md`).

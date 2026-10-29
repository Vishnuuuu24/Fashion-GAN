# Fashion GAN app

| Page | What it does |
|---|---|
| 🏠 Overview | Key numbers and how it works |
| 🎨 Generate | Generate new images, all categories, download, morph |
| 🔍 Input → Output | Inputs, output, closest real image, discriminator verdict |
| 📊 Results | Discriminator comparison, training, training log |

## Files
| Path | Contents |
|---|---|
| `models/fashion_generator.keras` | generator |
| `models/fashion_discriminator.keras` | cGAN discriminator |
| `models/fashion_judge.keras` | classifier that checks outputs |
| `models/real_samples.npz` | 300 real images per category |
| `assets/` | plots and logs from the notebook |

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

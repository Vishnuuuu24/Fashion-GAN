# 👗 Fashion GAN Studio — Streamlit app

| Page | What it does |
|---|---|
| 🏠 Overview | What the model is and how it works |
| 🎨 Generate | Pick a category (or all 10 with the same noise), new seed, judge's verdict per image, ZIP download, morph between two designs |
| 🔍 Input → Output | The noise + label going in, the image coming out, the closest real image and the difference, the discriminator's opinion |
| 📊 Results | Discriminator comparison, training animation and losses, and the log of every training run |

## Files
```
models/fashion_generator.keras       # the trained generator
models/fashion_discriminator.keras   # the trained cGAN discriminator
models/fashion_judge.keras           # classifier used to check generated images
models/real_samples.npz              # 300 real images per category (for "closest real image")
assets/                              # plots, gif and csv files from the notebook's gan_outputs/
```

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

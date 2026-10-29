# Study guide — Fashion GAN

Everything you need to explain the project, section by section, followed by likely questions and short answers.

---

## 1. The project in one paragraph
We built a **Conditional GAN** that generates new 28 × 28 grayscale clothing images of a chosen category, trained on
**Fashion-MNIST** (60,000 training + 10,000 test images, 10 categories). We wrote the generator, three different
discriminators and the training loop ourselves with TensorFlow/Keras (no pretrained model, no GAN library), compared
the three discriminators, trained the best one for 30 epochs, and showed how the **input** (noise + label) relates to
the **output** (image) and how generated images differ from real ones.

## 2. What is a GAN?
**Generative Adversarial Network** (Ian Goodfellow, 2014). Two networks compete:

| | Generator (G) | Discriminator (D) |
|---|---|---|
| Input | random noise z (+ label) | image (+ label) |
| Output | fake image | probability "real" (0–1) |
| Wants | D to say "real" for its fakes | to say 1 for real, 0 for fake |

Analogy: G = forger, D = detective. Each improves because of the other.

**Training loop (one batch):**
1. Take a batch of real images. Generate the same number of fakes with G.
2. **Update D**: loss = BCE(D(real), 1) + BCE(D(fake), 0).
3. **Update G**: loss = BCE(D(fake), 1) — G is rewarded when D thinks its fakes are real.

G never sees a real image; it learns only through D's gradients.

## 3. Data preparation
* `keras.datasets.fashion_mnist.load_data()` — downloads automatically.
* Pixels scaled from **[0, 255] to [-1, 1]** with `x / 127.5 - 1`, because G's last layer is **tanh** (range −1…1).
  Real and fake images must be in the same range or D could tell them apart by brightness alone.

## 4. Generator (G)
```
noise (100) ──Dense(7·7·128)──LeakyReLU──Reshape(7,7,128) ─┐
label (0–9) ──Embedding(10,50)──Dense(49)──Reshape(7,7,1) ─┴─ Concatenate → 7×7×129
   → Conv2DTranspose(128, stride 2) → BatchNorm → LeakyReLU      14×14
   → Conv2DTranspose(64,  stride 2) → BatchNorm → LeakyReLU      28×28
   → Conv2D(1, 7×7, tanh)                                         28×28×1 image
```
* **Noise z (100 numbers)** — random values from a normal distribution; give variety. Different z → different design.
* **Embedding** — turns the category number into a learned vector of 50 numbers (like a word embedding).
* **Conv2DTranspose** — "upsampling convolution": with stride 2 it doubles width and height.
* **BatchNormalization** — normalises activations per batch → more stable, faster training.
* **LeakyReLU(0.2)** — like ReLU but lets a small gradient through for negative inputs (avoids "dead" neurons).
* **tanh** output — pixel values in [-1, 1].

## 5. The three discriminators
All share: Conv2D(64, stride 2) → LeakyReLU → Dropout(0.3) → Conv2D(128, stride 2) → LeakyReLU → Dropout(0.3) → Flatten.
Stride 2 halves the size (28 → 14 → 7) instead of using pooling. **Dropout** stops D from memorising.

| | How the label is used | Output layer(s) | D loss |
|---|---|---|---|
| **Vanilla GAN** | not used at all | Dense(1) | BCE |
| **cGAN** | label → Embedding → Dense(784) → 28×28 map, **concatenated with the image as a 2nd channel** | Dense(1) | BCE |
| **AC-GAN** (Auxiliary Classifier GAN) | not an input — D must **predict** the class | Dense(1) real/fake **+** Dense(10) class | BCE + sparse categorical cross-entropy |

**Why compare them?**
* Vanilla: nobody checks the label → G ignores it → class accuracy ≈ 10 % (random). Images may look fine, but you
  can't choose the category. This shows *why* conditioning is needed.
* cGAN: D judges "is this a real image **of this category**?", so G must draw the right category.
* AC-GAN: G is also directly rewarded when D classifies its fake as the requested class → usually highest class
  accuracy, but can produce less variety inside a class.
* We chose **cGAN** for the final model: standard, simple, good balance of quality and variety.

**Logits:** the last Dense layer has no activation; it outputs a raw score (logit). `sigmoid(logit)` = probability.
We use `BinaryCrossentropy(from_logits=True)`, which applies the sigmoid inside the loss in a numerically stable way.

## 6. Losses and optimiser
* **Binary cross-entropy (BCE)**: `−[y·log(p) + (1−y)·log(1−p)]`. Small when the prediction p matches the target y.
* **D loss** = BCE(real → 1) + BCE(fake → 0).
* **G loss** = BCE(fake → 1) — the "non-saturating" form; gives G strong gradients even when D easily rejects fakes.
* **AC-GAN extra**: sparse categorical cross-entropy on the class prediction (for D on real and fake; for G on fake).
* **Adam**, learning rate **0.0002**, **beta_1 = 0.5** — the standard DCGAN settings (lower beta_1 = less momentum,
  helps the back-and-forth game stay stable).
* **Batch size 128.**

## 7. Training loop "from scratch"
We don't use `model.fit()` for the GAN. We write `train_step()` ourselves:
* `tf.GradientTape()` records operations so we can compute gradients of the loss.
* Step 1 applies gradients **only to D's weights**; step 2 **only to G's weights**.
* `@tf.function` compiles the step into a fast TensorFlow graph.
* `training=True` makes BatchNorm and Dropout behave in training mode.

## 8. How we measure results
* **Judge classifier** — a normal CNN trained on real images (≈ 90 % test accuracy). We generate images of each
  category and check how often the judge predicts the category we asked for = **class accuracy**. Random = 10 %.
* **Confusion matrix** — which categories get mixed up (shirt / T-shirt / pullover / coat — the judge confuses these
  on real images too).
* **Discriminator probability** — at a good balance D gives ≈ 0.5 to both real and generated images.
* **Losses** — GAN losses don't go to 0. Stable, level losses = a healthy fight. D loss → 0 means D is winning and G
  stops learning; G loss exploding means the same.

## 9. Input vs output (what the faculty asked)
* **Inputs:** 100 noise numbers (shown as a 10 × 10 colour grid) + a category label (one-hot). **Output:** 28 × 28 image.
* **Same noise, different labels** → different products in a similar style ⇒ the **label controls what** is drawn.
* **Same label, different noise** → different designs of the same product ⇒ the **noise controls the style**.
* **Average real vs average generated image** per category + their difference ⇒ G learned the typical shape.
* **Generated vs closest real training image** + difference map ⇒ the difference is clearly visible, so G is
  **creating new images, not copying** training images.
* **Untrained vs trained generator** ⇒ at the start the output is pure noise; after training it is clothing.

## 10. Training runs
Every run's models are saved (`models/runs/<run name>/`) and logged in `training_log.csv`, even runs that were worse.
See `TRAINING_LOG.md` for the notes.

---

## Likely questions and answers

**Q: What is a GAN?**
Two neural networks — a generator that creates fake data from random noise and a discriminator that tries to tell
real from fake — trained against each other until the generator's output looks real.

**Q: What is the difference between a GAN and a conditional GAN?**
A conditional GAN gives the class label to both networks, so we can choose which category is generated. A plain GAN
generates a random category.

**Q: What is the latent vector / noise vector?**
100 random numbers from a normal distribution, the generator's main input. Each different vector gives a different image.
It is called "latent" because its meaning (style, shape) is learned, not designed.

**Q: Why tanh in the generator's last layer?**
To produce pixel values in [-1, 1], which matches how we scaled the real images.

**Q: Why LeakyReLU instead of ReLU?**
ReLU outputs 0 (and gradient 0) for negative inputs, so neurons can "die". LeakyReLU keeps a small slope (0.2), so
gradients always flow — important because G learns only through D's gradients.

**Q: What does Conv2DTranspose do?**
The reverse of a strided convolution: it increases the spatial size (here doubles it: 7 → 14 → 28).

**Q: Why Embedding for the label?**
It turns a category number (0–9) into a learned vector, so the network can learn what each category "means".

**Q: Why is there no pooling in the discriminator?**
Strided convolutions (stride 2) downsample while learning how to do it; this is the DCGAN recommendation.

**Q: What does Dropout do in the discriminator?**
Randomly switches off 30 % of neurons during training, so D can't simply memorise the training images and become too
strong too quickly.

**Q: Why do we train D and G separately?**
They have opposite goals. When training D we freeze G (only D's weights change) and vice versa.

**Q: What loss do you use?**
Binary cross-entropy. D: real → 1, fake → 0. G: wants D to output 1 for its fakes. AC-GAN adds a categorical
cross-entropy for the class prediction.

**Q: What is `from_logits=True`?**
The discriminator outputs raw scores (logits). This tells the loss to apply the sigmoid itself, which is more
numerically stable.

**Q: Why Adam with lr = 0.0002 and beta_1 = 0.5?**
The standard settings from the DCGAN paper; they make GAN training more stable.

**Q: Why don't the losses go to zero?**
G and D compete — when one improves, the other's loss goes up. A healthy GAN has roughly level losses. If D's loss
goes to 0, D has won and G gets no useful gradient.

**Q: What is mode collapse?**
When G produces only a few kinds of images (e.g. always the same shoe) because they fool D. Conditioning on the label
and dropout in D help avoid it. (AC-GAN is a bit more prone to low variety.)

**Q: How do you know the generated images are good?**
(1) Visually. (2) A separately trained classifier (the judge, ~90 % on real images) recognises the requested category
in most generated images. (3) The discriminator gives similar probabilities to real and generated images.

**Q: How do you know the GAN is not just copying training images?**
For each generated image we found the closest real training image and showed the difference — they are clearly
different. Also, the same label with different noise gives many different designs.

**Q: Which discriminator worked best and why?**
See the comparison table in the notebook / app. Vanilla can't control the category (≈ 10 % class accuracy). cGAN and
AC-GAN both follow the label; AC-GAN usually has the highest class accuracy (G is directly rewarded for being
classifiable), cGAN gives more variety. We used cGAN as the final model.

**Q: What does the label do vs what does the noise do?**
Label = which product (category). Noise = style (shape, thickness, brightness, pattern). Shown in the "same noise,
different label" and "same label, different noise" grids.

**Q: Why Fashion-MNIST?**
Small, clean, labelled, 10 classes, quick to train on a free Colab GPU — good for learning. It is harder than digit
MNIST, so results are more interesting.

**Q: What are the limitations?**
Small 28 × 28 grayscale images; similar categories (shirt vs T-shirt) are sometimes mixed up; GAN training is
sensitive to settings.

**Q: How could you improve it?**
Train longer, use bigger models or higher-resolution colour datasets, or try newer methods (e.g. DCGAN on larger
images, StyleGAN, diffusion models).

**Q: What does "from scratch" mean in your project?**
All layers are defined by us with basic Keras layers, weights start random (no pretrained model), and the training
loop (losses, gradients, alternating updates) is written by hand with `tf.GradientTape` instead of a GAN library.

**Q: What is `@tf.function`?**
It compiles the Python training step into a TensorFlow graph, which runs much faster on the GPU.

**Q: What is the judge classifier?**
An ordinary CNN (2 conv layers + dense) trained on the real images. It's only used to *evaluate* the GAN.

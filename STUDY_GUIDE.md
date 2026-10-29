# Study guide

## 1. Project summary
A conditional GAN that generates 28×28 grayscale clothing images of a chosen category. Trained from scratch on
Fashion-MNIST. Three discriminators compared. Final cGAN: 89.3% class accuracy.

## 2. GAN basics
| | Generator (G) | Discriminator (D) |
|---|---|---|
| Input | noise z + label | image + label |
| Output | fake image | probability real |
| Goal | fool D | catch fakes |

One training step:
1. Train D: real → 1, fake → 0.
2. Train G: make D output 1 for fakes.

G never sees real images. It learns only through D's gradients.

## 3. Data
* `keras.datasets.fashion_mnist`: 60,000 train, 10,000 test.
* Scaled to [-1, 1] to match G's `tanh` output.

## 4. Generator
```
noise (100) → Dense(7·7·128) → LeakyReLU → Reshape(7,7,128) ─┐
label       → Embedding(10,50) → Dense(49) → Reshape(7,7,1) ─┴→ Concatenate (7×7×129)
→ Conv2DTranspose(128, stride 2) → BatchNorm → LeakyReLU   14×14
→ Conv2DTranspose(64, stride 2)  → BatchNorm → LeakyReLU   28×28
→ Conv2D(1, 7×7, tanh)                                      28×28×1
```
| Layer | Purpose |
|---|---|
| Embedding | label → learned vector |
| Conv2DTranspose | upsamples (doubles size) |
| BatchNorm | stable training |
| LeakyReLU | keeps gradients for negative inputs |
| tanh | output in [-1, 1] |

## 5. Discriminators
Shared body: Conv2D(64, stride 2) → LeakyReLU → Dropout(0.3) → Conv2D(128, stride 2) → LeakyReLU → Dropout(0.3) → Flatten.

| | Label use | Output | Loss |
|---|---|---|---|
| Vanilla | none | real/fake | BCE |
| cGAN | label → 28×28 map, added as 2nd channel | real/fake | BCE |
| AC-GAN | predicts the label | real/fake + class | BCE + cross-entropy |

D outputs a logit. `sigmoid(logit)` = probability. Losses use `from_logits=True` for stability.

## 6. Losses and optimizer
* BCE = −[y·log(p) + (1−y)·log(1−p)]
* D loss = BCE(real, 1) + BCE(fake, 0)
* G loss = BCE(fake, 1)
* AC-GAN adds class cross-entropy to both.
* Adam, lr 0.0002, beta_1 0.5, batch 128.

## 7. Training loop
* `tf.GradientTape` records operations to compute gradients.
* Step 1 updates only D. Step 2 updates only G.
* `@tf.function` compiles the step for speed.
* `training=True` enables BatchNorm and Dropout training mode.

## 8. Evaluation
* **Judge:** CNN trained on real images, 90.3% accurate.
* **Class accuracy:** % of generated images the judge labels correctly. Random = 10%.
* **Losses:** should stay level. D loss → 0 means D won and G stopped learning.

## 9. Input vs output
* Inputs: 100 noise values + label. Output: 28×28 image.
* Same noise, different labels → label controls **what**.
* Same label, different noise → noise controls **style**.
* Average real vs generated → G learned each category's shape.
* Closest real image + difference → outputs are new, not copies.

## 10. Results
| | Class accuracy |
|---|---|
| Vanilla (10 ep) | 10.1% |
| cGAN (10 ep) | 81.6% |
| AC-GAN (10 ep) | 94.9% |
| Final cGAN (30 ep) | 89.3% |

---

## Viva Q&A

**What is a GAN?**
Two networks: a generator makes fakes from noise, a discriminator separates real from fake. They train against each other.

**GAN vs conditional GAN?**
A conditional GAN also receives the label, so you choose the category.

**What is the noise vector?**
100 random numbers from a normal distribution. Different noise gives a different image.

**Why tanh?**
Outputs [-1, 1], matching the scaled real images.

**Why LeakyReLU?**
ReLU gives zero gradient for negative inputs. LeakyReLU keeps a small slope, so gradients keep flowing.

**What does Conv2DTranspose do?**
Upsamples. With stride 2 it doubles width and height.

**Why Embedding for the label?**
Turns a category number into a learned vector the network can use.

**Why no pooling in D?**
Stride-2 convolutions downsample and learn how to do it (DCGAN guideline).

**Why Dropout in D?**
Stops D from memorising and overpowering G too early.

**Why train D and G separately?**
They have opposite goals. Each step updates one and freezes the other.

**What loss is used?**
Binary cross-entropy. AC-GAN adds categorical cross-entropy.

**What is `from_logits=True`?**
D outputs raw scores. The loss applies sigmoid internally, which is more stable.

**Why lr 0.0002 and beta_1 0.5?**
Standard DCGAN settings for stable GAN training.

**Why don't the losses reach zero?**
G and D compete. When one improves, the other's loss rises. Level losses = healthy training.

**What is mode collapse?**
G produces only a few kinds of images. Labels and Dropout in D help prevent it.

**How do you know the images are good?**
Visual check, 89.3% class accuracy (judge on real: 90.3%), and D scores real and fake similarly.

**How do you know it doesn't copy?**
Each output differs clearly from its closest training image. Different noise gives different designs.

**Which discriminator was best?**
AC-GAN had the highest accuracy (94.9%) but less variety. cGAN (81.6%) is the standard choice and was used for the final model.

**Why did vanilla get 10%?**
Its D ignores the label, so G learns to ignore it too. Same as random guessing.

**What changed between runs?**
First the discriminator (3 runs, 10 epochs each), then epochs for cGAN (10 → 30, accuracy 81.6% → 89.3%).

**Why does D give ~0.4 to real and generated images?**
G fools D most of the time. The ideal is 0.5 for both. Noise also scores ~0.4 because D never saw noise in training.

**Why is the worst category Shirt?**
Shirts look like T-shirts, pullovers and coats. The judge gets real shirts only 60.9% right.

**Why grayscale, not RGB?**
Fashion-MNIST is grayscale. Change the output to 3 channels and train on a colour dataset (e.g. CIFAR-10) for RGB.

**How is this different from modern AI image tools?**
Those mostly use diffusion models with text prompts, billions of images and huge compute. The core idea is the same: noise → image.

**What does "from scratch" mean here?**
All layers defined by us, random initial weights, hand-written training loop with `tf.GradientTape`. No pretrained model or GAN library.

**What is `@tf.function`?**
Compiles the Python training step into a fast TensorFlow graph.

**Limitations?**
Small grayscale images. Similar tops get mixed up. Sensitive to settings.

**Improvements?**
Train longer, bigger models, colour datasets, newer methods (StyleGAN, diffusion).

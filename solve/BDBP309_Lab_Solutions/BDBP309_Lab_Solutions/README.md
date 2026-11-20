# BDBP-309 Deep Learning Laboratory — Solved Practical Reference

This folder is organised around the **BDBP-309 Deep Learning Laboratory handout (20 July 2026)**. It contains separate runnable Python scripts for the practical exercises from Labs 1–21, including the additional-material exercises listed in the handout.

## Folder contents

| Handout lab | File(s) | What it demonstrates |
|---|---|---|
| 1. Activation functions | `01_activation_functions.py` | Sigmoid, tanh, ReLU, leaky ReLU, softmax and derivatives from scratch; plots |
| 2. Forward pass | `02_forward_pass.py` | Single neuron and multilayer network; vectorised forward pass |
| 3. Backpropagation | `03_backpropagation.py` | Chain rule, gradients and one SGD update |
| 4. One-layer network from scratch | `04_one_layer_nn_from_scratch.py` | Manual forward/backward training and loss plot |
| 5. XOR from scratch | `05_xor_from_scratch.py` | Two-layer neural network with manual backpropagation |
| 6. PyTorch basics | `06_pytorch_basics.py` | Tensors, Dataset, DataLoader, autograd, model, optimizer, save/load |
| 7. Feedforward CPU/GPU | `07_feedforward_classifier.py` | End-to-end classifier with automatic CPU/GPU selection |
| 8. Normalization, dropout, optimizers | `08_batchnorm_dropout_optimizers.py` | BatchNorm and LayerNorm from scratch, dropout, SGD, Momentum, Adagrad and Adam |
| 9. Vanishing/exploding gradients | `09_vanishing_exploding_gradients.py` | Gradient-product simulation and log plot |
| 10. Landmark-to-target gene prediction | `10_landmark_gene_prediction.py` | Tabular expression input and PyTorch regression network |
| 11. CNN from scratch | `11_cnn_from_scratch.py` | 3×3 convolution and 2×2 max pooling with loops |
| 12. CNN applications | `12_cnn_mnist_cifar10.py` | MNIST or CIFAR10 CNN; compares different depths |
| 13. Gene accessibility CNN | `13_gene_accessibility_cnn.py` | DNA one-hot representation and 1D CNN classifier |
| 14. Vanilla RNN | `14_vanilla_rnn_from_scratch.py` | RNN forward pass and backpropagation through time using NumPy |
| 15. Image captioning | `15_rnn_image_caption.py`, `15_lstm_image_caption.py` | CNN image encoder with RNN and LSTM caption decoders |
| 16. Sequence-to-sequence | `16_seq2seq_lstm_gru.py` | Encoder-decoder translation with LSTM and GRU |
| 17. Protein function prediction | `17_protein_function_rnn.py` | Amino-acid encoding and bidirectional GRU multilabel classifier |
| 18. Embeddings | `18_representation_learning.py` | Text TF-IDF/SVD, image PCA, biological-style PCA and autoencoder embeddings |
| 19. GAN and Stable Diffusion | `19_gan_image_caption.py`, `19_stable_diffusion_caption.py` | Adversarial caption generator; BLIP captioning with optional Stable Diffusion image generation |
| 19. Transformer (additional) | `19_transformer_image_caption.py` | Vision Transformer + GPT-2 image captioning model |
| 20. Self-supervised learning (additional) | `20_self_supervised_learning.py` | SimCLR contrastive image representation learning followed by a linear-probe classifier |
| 21. Foundation models (additional) | `21_medical_foundation_finetuning.py` | Fine-tunes a BiomedCLIP visual encoder on a custom medical image folder |

Also included:

- `data_loading_reference.py` — commented examples for CSV, Excel, JSON, JSONL, Parquet, NPY, NPZ, text, FASTA, images, ImageFolder, PyTorch tensors and custom Dataset/DataLoader. It shows common feature/label splits (`X` and `y`).
- `requirements.txt` — core packages.
- `optional_requirements.txt` — packages for captioning, diffusion and foundation-model scripts.
- `sample_data/` — tiny tabular and FASTA files to test loading examples. These are illustrative data, not biological measurements.

## Setup

Use a virtual environment. In Windows PowerShell:

```powershell
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For optional pretrained model scripts:

```powershell
pip install -r optional_requirements.txt
```

For CPU-only PyTorch, install the CPU build recommended by the official PyTorch selector for your system, then install the remaining packages from `requirements.txt` without replacing your PyTorch installation.

## Running a practical

Run commands from this folder, for example:

```powershell
python 01_activation_functions.py
python 07_feedforward_classifier.py
python 18_representation_learning.py
```

Some scripts download public datasets or pretrained weights on the first run. `12_cnn_mnist_cifar10.py` defaults to MNIST; change `vin_dataset_name` to `"CIFAR10"` for CIFAR10. The code uses small subsets and a few epochs for practical-class demonstrations. Increase the subset size and epochs for proper experiments.

## Supplying your own data

- **Tabular classification/regression:** CSV or Excel with one row per sample. The code comments show how to select feature columns into `vin_x` and target column into `vin_y`.
- **Landmark gene prediction:** provide `gene_expression.csv`; numeric columns are features and `target_gene` is the regression target. The fallback data is explicitly synthetic.
- **Gene accessibility:** provide `accessibility.csv` with `sequence,label`; sequence encoding is one-hot A/C/G/T.
- **Image captioning / GAN:** provide `captions.csv` with `image,caption` and place the matching files inside `images/`.
- **Seq2seq:** provide `translation_pairs.csv` with `source,target`.
- **Protein function:** provide `protein_functions.csv` with `sequence,functions`; separate multiple functions with semicolons, e.g. `binding;enzyme`.
- **Medical image fine-tuning:** place images in class-named subfolders under `medical_images/`, e.g. `medical_images/class_0/*.png` and `medical_images/class_1/*.png`.

For medical imaging, split at the patient level where applicable to avoid data leakage. The included random split is only a compact practical template; it is not a substitute for a study-grade validation protocol.

## Important interpretation notes

1. **Synthetic fallbacks are for code execution only.** They do not represent real biological findings and must be replaced with the lab's assigned/public datasets for analysis.
2. The handout names a target-gene dataset and papers but its extracted PDF does not expose a direct downloadable dataset URL. The target-gene script therefore accepts a generic expression CSV and uses a synthetic fallback until the assigned dataset is supplied.
3. **Stable Diffusion is text-to-image, not image-to-caption.** The captioning script uses BLIP to convert an image into text. It optionally uses Stable Diffusion to generate a new image from that caption. This avoids describing a text-to-image model as an image captioner.
4. The GAN script uses a compact educational conditional caption generator and a discriminator. Gumbel-Softmax provides a differentiable approximation for word selections; it is intended to demonstrate the adversarial setup, not to compete with pretrained captioning models.
5. The foundation-model practical uses Microsoft's BiomedCLIP through OpenCLIP. It expects a labelled custom medical image folder and downloads model weights on first use.
6. For the captioning and pretrained-model scripts, internet access may be required for model checkpoint downloads. The Stable Diffusion generation step is disabled by default because it is compute- and memory-intensive.

## Notes on code style

The scripts use explicit functions, loops, visible intermediate variables (usually prefixed with `vin_`), comments, and direct print statements to make the practical steps easy to follow during revision.

# PyTorch Practical Activity — Framework Essentials

A hands-on notebook covering the fundamental building blocks of PyTorch, from tensors to training a CNN on MNIST.

---

## Contents

| Section | Topic |
| :---: | :--- |
| **0** | Environment setup (CPU / CUDA detection) |
| **1** | General workflow overview |
| **2** | Tensors & manipulation |
| **3** | CPU / GPU and `device` |
| **4** | Autograd (automatic differentiation) |
| **5** | A single neuron with `nn.Linear` |
| **6** | Building networks with `nn.Module` |
| **7** | Activation functions (ReLU, tanh, sigmoid) |
| **8** | Forward propagation |
| **9** | Loss functions (MSE, CrossEntropy, BCE) |
| **10** | Optimizers (SGD, Adam) |
| **11** | The training loop |
| **12** | `train()` vs `eval()` |
| **13** | `Dataset` / `DataLoader` |

### Practical exercises

| # | Task | Dataset | Result |
| :---: | :--- | :--- | :--- |
| **1** | Salary regression | Synthetic | MAE ≈ 387 DH |
| **2** | IRIS classification | IRIS (150 samples) | 100 % accuracy |
| **3** | MNIST with MLP | MNIST | 97.74 % accuracy |
| **4** | MNIST with CNN | MNIST | 98.84 % accuracy |

---

## Getting started

### Requirements

```text
python >= 3.10
torch >= 2.0
torchvision
torchmetrics
torcheval
scikit-learn
matplotlib
numpy
```

### Installation

```bash
# Clone the repo
git clone "myrepo"

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate       # Windows

```

### Running the notebook

The notebook works on CPU but will automatically use CUDA if available.

---

## Project structure

```text
.
├── Activité pratique Bases de Pytorch.ipynb   # Main notebook
├── report.md                                   # Detailed report of all exercises
├── README.md                                   # This file
├── requirements.txt
└── data/                                       # MNIST data (auto-downloaded)
```

---

## What you'll learn

* **Tensors :** creation, reshaping, indexing, broadcasting, and matrix operations.
* **Autograd :** deep dive into `requires_grad` and how `.backward()` computes gradients.
* **Modules :** utilizing `nn.Linear`, `nn.Sequential`, and writing custom `nn.Module` subclasses.
* **Training Loop :** master the invariant sequence: `zero_grad` → `forward` → `loss` → `backward` → `step`.
* **Evaluation :** proper utilization of `model.eval()` and the `torch.no_grad()` context manager.
* **Metrics :** tracking performance via accuracy, recall, precision, F1-score, and confusion matrices.
* **Saving :** implementing `.state_dict()` saving and loading best practices.

---

## Results summary

| Exercise | Model | Metric | Value |
| :--- | :--- | :---: | :--- |
| **Salary** | Linear(2→1) | MAE | 387 DH |
| **IRIS** | Linear(4→16→3) | Accuracy | 100 % |
| **IRIS baseline** | Linear(4→3) | Accuracy | 90 % |
| **MNIST** | MLP 784→128→10 | Accuracy | 97.74 % |
| **MNIST** | CNN 2×(Conv+Pool)+FC | Accuracy | 98.84 % |
| **MNIST** | CNN + Dropout 0.5 | Accuracy | 98.64 % |

---

## Common pitfalls

| Pitfall | Fix |
| :--- | :--- |
| Forgetting `optimizer.zero_grad()` | Gradients will accumulate across training steps, ruining updates. |
| Using `y * 1000` output without conversion | Always restore original units before interpreting your metrics. |
| Adding Softmax before `CrossEntropyLoss` | `CrossEntropyLoss` already processes raw logits (includes implicit softmax). |
| Fitting `StandardScaler` on the test set | Avoids data leakage — fit your tools on the training partition only. |
| Saving the full model object | Save only the weights using `state_dict()` to prevent serialization issues. |
| Evaluating with gradients turned on | Always wrap your inference tracking loops inside `with torch.no_grad():`. |

---

## Extending the exercises

* Increase CNN depth by appending additional VGG-style blocks or adding `BatchNorm2d` layers.
* Experiment with lower dropout rates (`Dropout(p=0.3)`) paired with a higher number of training epochs.
* Replace the `Adam` optimizer with traditional `SGD` enhanced by momentum tuning.
* Introduce data augmentation policies inside the pipeline via `RandomRotation` or `RandomAffine`.
* Port the training loop implementation to other challenging benchmarks like `Fashion-MNIST` or `CIFAR-10`.

---

## Author

* **Name :** Bounjoume Oussama
* **Contact :** <bounjoum.noujoum@gmail.com>

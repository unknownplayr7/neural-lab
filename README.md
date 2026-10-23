# Neural Lab

An interactive image-classification workbench built with **PyTorch** and **Streamlit**. Train a convolutional neural network on CIFAR-10, inspect learning curves, evaluate generalization, and try predictions on uploaded images.

> **Scope:** the model is trained for CIFAR-10's ten classes. Predictions on unrelated images are exploratory, not general-purpose object recognition.

## Features

- **Apple Silicon friendly:** device auto-detection for NVIDIA CUDA, Apple MPS, and CPU fallback.
- **Controlled experiments:** epochs, batch size, learning rate, random seed, dataset subset size, and augmentation.
- **Visible learning:** live batch progress, epoch metrics, loss curves, and train/test accuracy curves.
- **Evaluation:** held-out test metrics without gradient updates.
- **Prediction sandbox:** upload a photo and inspect the top five CIFAR-10 class probabilities.
- **Portable checkpoints:** save model weights and metadata, then reload them from the app.
- **Quality baseline:** reusable training code, unit tests, and GitHub Actions CI.

## Quick start

Python 3.11 is recommended. On macOS, create an isolated environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

The first dataset use downloads CIFAR-10 into `data/`. The app downloads it only when training or evaluation is requested. For Apple Silicon, install a recent PyTorch build with MPS support; the app automatically falls back to CPU if no accelerator is available.

## Suggested workflow

1. Open **Train** and start with 5 epochs and 10,000 training examples to get a quick baseline.
2. Watch the per-batch progress and epoch-level loss/accuracy curves.
3. Increase the sample limit to 50,000 and train longer once the pipeline works.
4. Use **Evaluate** to measure performance on the selected held-out test subset.
5. Use **Predict** to inspect top-five class scores for an uploaded image.
6. Save a checkpoint before experimenting with different settings.

CIFAR-10 labels are: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.

## Tests and development

```bash
python -m pytest
streamlit run app.py
```

GitHub Actions runs the unit test suite on Python 3.10, 3.11, and 3.12 for pushes to `main` and feature branches, pull requests, and manual runs.

## Project layout

```text
app.py                    Streamlit UI
neural_lab/model.py       CNN, device selection, checkpoint helpers
neural_lab/data.py        CIFAR-10 transforms and data loaders
neural_lab/training.py    Reusable train/evaluate loops
tests/                    Unit tests
.github/workflows/ci.yml  Continuous integration
```

## Model and data notes

- The CNN uses convolutional blocks, batch normalization, ReLU, max pooling, dropout, adaptive average pooling, and a ten-class linear head.
- The loss is cross-entropy and the optimizer is AdamW.
- A fixed seed is applied to Python, NumPy, and PyTorch when a training run starts. Accelerator kernels can still introduce small nondeterministic differences.
- Test accuracy is an estimate for the held-out CIFAR-10 test split. Repeated tuning against it can make it less independent.
- Uploaded images are converted to RGB, resized to 32×32, and normalized like CIFAR-10. High model confidence is not proof that a prediction is correct, especially for out-of-distribution images.
- Training runs inside the Streamlit process. Closing or refreshing the app may interrupt the run.
- Checkpoints should only come from trusted sources. The app uses weights-only deserialization when supported by the installed PyTorch version.

## License

MIT. See [LICENSE](LICENSE).

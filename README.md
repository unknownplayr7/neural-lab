# Neural Lab

An interactive image-classification workbench built with **PyTorch** and **Streamlit**. Train a CNN from scratch on CIFAR-10 or CIFAR-100, inspect live learning curves, examine misclassifications, evaluate generalization, and try predictions on uploaded images.

> **Scope:** the model predicts among the selected dataset's known classes. Predictions on unrelated images are exploratory, not general-purpose object recognition.

## Features

- **Streaming data:** examples are read from Hugging Face Datasets as batches are iterated. The full CIFAR archive is not downloaded and stored on your machine.
- **CIFAR-10 and CIFAR-100:** choose a 10-class or 100-class recognition task.
- **Apple Silicon friendly:** automatic CUDA, Apple MPS, and CPU device selection.
- **Controlled experiments:** epochs, batch size, learning rate, random seed, sample limits, and augmentation.
- **Visible learning:** live batch progress, epoch metrics, train/validation curves, and held-out test metrics.
- **Evaluation:** confusion matrix and per-class accuracy on a separate test split.
- **Prediction sandbox:** upload a photo and inspect the top class probabilities.
- **Pause and resume:** train one epoch at a time or run the remaining epochs; checkpoints preserve optimizer state and metric history.
- **Error analysis:** inspect a confusion matrix, per-class accuracy, and a gallery of misclassified test images.
- **Portable checkpoints:** save model weights and metadata, then reload them.
- **Quality baseline:** reusable training code, tests, and GitHub Actions CI.

## Quick start

Python 3.11 is recommended. On macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

A network connection is required for streamed examples. The Hugging Face dataset library may cache small metadata/configuration files, but Neural Lab does not download the full dataset archive into a local `data/` folder. Streaming can be slower than local cached data, and later epochs may read examples from the remote source again.

## Suggested workflow

1. Start with CIFAR-10, 5 epochs, and 10,000 training examples for a quick baseline.
2. Watch batch progress and the train-versus-validation curves.
3. Increase the sample cap or switch to CIFAR-100 for a harder 100-class problem.
4. Use **Evaluate** to inspect the held-out test confusion matrix and per-class accuracy.
5. Use **Predict** to inspect the top five class scores for an uploaded image.
6. Save a checkpoint before experimenting with new settings.

## Tests and development

```bash
python -m pytest
streamlit run app.py
```

GitHub Actions runs tests on Python 3.10, 3.11, and 3.12 for pull requests, pushes to `main`, and manual runs.

## Project layout

```text
app.py                    Streamlit UI
neural_lab/model.py       CNN, device selection, checkpoint helpers
neural_lab/data.py        Hugging Face streaming pipeline and transforms
neural_lab/training.py    Reusable train/evaluate loops
tests/                    Unit tests
.github/workflows/ci.yml  Continuous integration
```

## Model and data notes

- The model is trained from random initialization, not a pretrained image classifier.
- The CNN uses convolutional blocks, batch normalization, ReLU, max pooling, dropout, adaptive average pooling, and a linear class head.
- Training uses cross-entropy and AdamW. Training augmentation uses random crop and horizontal flip.
- Training and validation consume disjoint windows from a seeded shuffled training stream. The separate official test split is reserved for final evaluation.
- Test accuracy is an estimate for the selected test subset. Repeated tuning against it can make it less independent.
- Uploaded images are converted to RGB, resized to 32×32, and normalized using dataset-specific statistics. High confidence is not proof of correctness for out-of-distribution images.
- MPS support depends on the installed PyTorch/macOS combination. CPU remains a fallback.
- Training runs inside the Streamlit process. Closing or refreshing the app may interrupt the run.
- Only load checkpoints from trusted sources. Neural Lab uses weights-only deserialization when supported by PyTorch.

## License

MIT. See [LICENSE](LICENSE).

# Neural Lab

Neural Lab is a public, open-source, **macOS-only** image-classification workbench. It is designed for people learning how convolutional neural networks train, evaluate, and fail. It uses PyTorch and Streamlit, with Apple Silicon acceleration through Metal Performance Shaders (MPS) when available.

This project intentionally targets macOS. It is not advertised as cross-platform software and does not include Windows or Linux support.

## What it does

- Train a convolutional neural network from random initialization on CIFAR-10 or CIFAR-100.
- Stream examples from Hugging Face rather than downloading full dataset archives into a local data directory.
- Inspect batch progress, training and validation metrics, learning curves, confusion matrices, per-class accuracy, and representative mistakes.
- Evaluate against a separate held-out test split.
- Upload an image and inspect the model's top class probabilities.
- Save and resume checkpoints containing model weights, optimizer and scheduler state, early-stopping progress, metric history, and experiment settings.
- Reduce the learning rate automatically when validation loss plateaus, and optionally stop training after a configurable patience window.

**Scope:** this is an educational classifier, not a general-purpose object detector. It predicts only among the selected dataset's classes. An uploaded photo outside those classes may receive a confident but incorrect prediction.

## Platform requirements

- macOS 13 or later recommended.
- Apple Silicon Mac recommended for MPS acceleration. Intel Macs can use CPU mode but will be substantially slower.
- Python 3.11 recommended.
- Internet access for streamed dataset examples.
- Several GB of free memory and storage recommended for dependencies and training.

MPS availability depends on the macOS and PyTorch versions installed. The application falls back to CPU when MPS is unavailable. Neural Lab does not support Windows or Linux.

## Quick start

Open Terminal:

```bash
git clone https://github.com/unknownplayr7/neural-lab.git
cd neural-lab
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the local address printed by Streamlit in your browser. The first run needs internet access to retrieve dataset metadata and stream examples. Hugging Face and its upstream dataset repositories are external services; availability and speed can vary.

## Suggested first experiment

1. Select CIFAR-10.
2. Start with 5 epochs and 10,000 training examples.
3. Use batch size 64 and the default learning rate.
4. Leave the plateau scheduler enabled to reduce the learning rate when validation loss stalls. Enable early stopping when you want a run to stop after several non-improving epochs.
5. Train one epoch at a time and inspect the learning curves and recorded learning rates.
6. Evaluate after training; avoid repeatedly tuning against the held-out test subset.
7. Save a checkpoint before changing experiment settings.

CIFAR-100 is a more difficult 100-class task and may require more training time and tuning.

## Development and tests

```bash
python -m pip install -r requirements.txt
python -m pytest
```

GitHub Actions runs the test suite on macOS. Hosted CI is not a substitute for testing MPS performance on each supported Mac model.

## Project layout

```text
app.py                    Streamlit user interface
neural_lab/model.py       CNN, device selection, checkpoint helpers
neural_lab/data.py        Hugging Face streaming pipeline and transforms
neural_lab/training.py    Reusable training and evaluation loops
tests/                    Automated tests
.github/workflows/ci.yml  Continuous integration
```

## Data and privacy

- Dataset examples are streamed from Hugging Face while training or evaluating. The dataset library may cache metadata and small files.
- Uploaded images and checkpoints are processed by the local Streamlit process. They are not intentionally sent to a Neural Lab server because the project has no hosted backend.
- Do not upload images you do not have permission to use.
- Checkpoints should come from trusted sources. Neural Lab uses PyTorch weights-only loading and rejects unsupported checkpoint formats.
- Keep the Streamlit server bound to localhost unless you understand the network exposure and have secured the environment.

## Known limitations

- Training runs in the Streamlit process. Closing the app or interrupting the process stops the current run.
- Remote streaming can make later epochs slow because examples may be read from the upstream dataset again.
- CIFAR images are only 32 by 32 pixels. Uploaded photos are resized to this resolution, which discards detail.
- Confidence scores are not calibrated probabilities of real-world correctness.
- Performance varies by Mac, memory capacity, PyTorch build, and network performance.

## Security

See [SECURITY.md](SECURITY.md) for the security policy.

## License

MIT. See [LICENSE](LICENSE).
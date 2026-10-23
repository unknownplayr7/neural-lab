"""Streamlit front end for Neural Lab."""
from __future__ import annotations

import io

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageOps
from torch import nn

from neural_lab.data import make_loaders
from neural_lab.model import CIFAR10_CLASSES, CIFAR10_MEAN, CIFAR10_STD, CifarCNN, available_device, save_checkpoint
from neural_lab.training import run_epoch, seed_everything, train_model

st.set_page_config(page_title="Neural Lab", page_icon="🧠", layout="wide")
st.markdown("""
<style>
.block-container {max-width: 1400px; padding-top: 1.8rem; padding-bottom: 3rem;}
.hero {padding: 1.4rem 1.6rem; border: 1px solid rgba(128,128,128,.25);
       border-radius: 18px; background: linear-gradient(120deg, rgba(92,124,250,.13), rgba(32,201,151,.08));}
.muted {color: var(--text-color); opacity: .72;}
div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.22); padding: .8rem 1rem; border-radius: 12px;}
</style>
<div class="hero">
  <h1>🧠 Neural Lab</h1>
  <p>Train it. Watch it learn. Find out where it gets confused.</p>
  <p class="muted">A hands-on CIFAR-10 image-classification workbench powered by PyTorch.</p>
</div>
""", unsafe_allow_html=True)

if "model" not in st.session_state:
    st.session_state.model = None
if "history" not in st.session_state:
    st.session_state.history = []
if "last_eval" not in st.session_state:
    st.session_state.last_eval = None


def device_options() -> list[str]:
    options = ["auto", "cpu"]
    if torch.cuda.is_available():
        options.append("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        options.append("mps")
    return options


@st.cache_resource(show_spinner=False)
def get_loaders(batch_size: int, train_limit: int, test_limit: int, augment: bool, seed: int):
    return make_loaders(
        root="data", batch_size=batch_size,
        train_limit=None if train_limit == 0 else train_limit,
        test_limit=None if test_limit == 0 else test_limit,
        augment=augment, num_workers=0, seed=seed,
    )


def prepare_image(image: Image.Image) -> torch.Tensor:
    image = ImageOps.exif_transpose(image).convert("RGB").resize((32, 32), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - np.asarray(CIFAR10_MEAN, dtype=np.float32)) / np.asarray(CIFAR10_STD, dtype=np.float32)
    return torch.from_numpy(array.copy()).permute(2, 0, 1).unsqueeze(0)


def show_history(history: list[dict[str, float]]) -> None:
    if not history:
        st.info("No training history yet. Start a run to populate the charts.")
        return
    rows = [{k: v for k, v in row.items() if k != "seconds"} for row in history]
    st.dataframe(rows, use_container_width=True, hide_index=True)
    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.plot([r["epoch"] for r in history], [r["train_loss"] for r in history], marker="o", label="Train")
        ax.plot([r["epoch"] for r in history], [r["test_loss"] for r in history], marker="o", label="Test")
        ax.set(title="Loss by epoch", xlabel="Epoch", ylabel="Cross-entropy loss")
        ax.grid(alpha=.25)
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with right:
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.plot([r["epoch"] for r in history], [100*r["train_accuracy"] for r in history], marker="o", label="Train")
        ax.plot([r["epoch"] for r in history], [100*r["test_accuracy"] for r in history], marker="o", label="Test")
        ax.set(title="Accuracy by epoch", xlabel="Epoch", ylabel="Accuracy (%)", ylim=(0, 100))
        ax.grid(alpha=.25)
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


with st.sidebar:
    st.header("Experiment setup")
    device_choice = st.selectbox("Compute device", device_options(), index=0, help="Auto prefers CUDA, then Apple Silicon MPS, then CPU.")
    epochs = st.slider("Epochs", min_value=1, max_value=30, value=5)
    batch_size = st.select_slider("Batch size", options=[16, 32, 64, 128, 256], value=64)
    learning_rate = st.select_slider("Learning rate", options=[0.0001, 0.0003, 0.001, 0.003, 0.01], value=0.001, format_func=lambda x: f"{x:g}")
    train_limit = st.select_slider("Training examples", options=[1000, 2500, 5000, 10000, 20000, 50000], value=10000, help="Full CIFAR-10 training uses 50,000 examples.")
    test_limit = st.select_slider("Evaluation examples", options=[1000, 2500, 5000, 10000], value=5000)
    seed = st.number_input("Random seed", min_value=0, max_value=2_147_483_647, value=42, step=1)
    augment = st.toggle("Random crop + flip augmentation", value=True)
    st.caption("Dataset downloads on first use. Training runs locally in this Streamlit process.")

try:
    device = available_device(device_choice)
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()

tab_train, tab_evaluate, tab_predict, tab_about = st.tabs(["🚀 Train", "📊 Evaluate", "🖼️ Predict", "🔬 Model & notes"])

with tab_train:
    st.subheader("Train a convolutional neural network")
    c1, c2, c3 = st.columns(3)
    c1.metric("Device", str(device).upper())
    c2.metric("Training samples", f"{train_limit:,}")
    c3.metric("Batch size", batch_size)
    st.write("Watch training and held-out test metrics to spot underfitting or overfitting.")

    if st.button("Start training", type="primary", use_container_width=True):
        seed_everything(int(seed))
        model = CifarCNN().to(device)
        st.session_state.model = model
        st.session_state.history = []
        st.session_state.last_eval = None
        try:
            with st.spinner("Preparing CIFAR-10. The first run may download the dataset…"):
                train_loader, test_loader = get_loaders(batch_size, train_limit, test_limit, augment, int(seed))
            progress = st.progress(0, text="Preparing training run…")
            live = st.empty()
            batch_progress = st.progress(0, text="Waiting for first batch…")
            epoch_rows: list[dict[str, float]] = []

            def batch_callback(batch_index: int, batch_count: int, rolling_loss: float) -> None:
                batch_progress.progress(batch_index / max(batch_count, 1), text=f"Batch {batch_index}/{batch_count} · running loss {rolling_loss:.3f}")

            def epoch_callback(epoch: int, train_metrics: dict[str, float], test_metrics: dict[str, float]) -> None:
                record = {
                    "epoch": float(epoch), "train_loss": train_metrics["loss"],
                    "train_accuracy": train_metrics["accuracy"], "test_loss": test_metrics["loss"],
                    "test_accuracy": test_metrics["accuracy"], "seconds": train_metrics["seconds"],
                }
                epoch_rows.append(record)
                st.session_state.history = list(epoch_rows)
                progress.progress(epoch / epochs, text=f"Completed epoch {epoch}/{epochs}")
                with live.container():
                    a, b, c, d = st.columns(4)
                    a.metric("Train loss", f"{train_metrics['loss']:.3f}")
                    b.metric("Train accuracy", f"{train_metrics['accuracy']:.1%}")
                    c.metric("Test loss", f"{test_metrics['loss']:.3f}")
                    d.metric("Test accuracy", f"{test_metrics['accuracy']:.1%}")
                    show_history(epoch_rows)

            train_model(model, train_loader, test_loader, device=device, epochs=epochs,
                        learning_rate=float(learning_rate), on_epoch=epoch_callback, on_batch=batch_callback)
            batch_progress.empty()
            st.success("Training finished. Review the curves, then evaluate or try an uploaded image.")
        except Exception as exc:
            st.error(f"Training failed: {exc}")
            st.exception(exc)
    show_history(st.session_state.history)
    if st.session_state.model is not None:
        st.divider()
        st.subheader("Save your model")
        checkpoint_name = st.text_input("Checkpoint filename", value="checkpoints/cifar10-cnn.pt")
        if st.button("Save checkpoint"):
            try:
                latest = st.session_state.history[-1] if st.session_state.history else {}
                saved = save_checkpoint(checkpoint_name, st.session_state.model,
                                        epoch=int(latest.get("epoch", 0)),
                                        accuracy=latest.get("test_accuracy"))
                st.success(f"Saved checkpoint to {saved}")
            except (OSError, ValueError) as exc:
                st.error(f"Could not save checkpoint: {exc}")

with tab_evaluate:
    st.subheader("Held-out evaluation")
    st.write("Evaluation uses the CIFAR-10 test split and does not update model weights.")
    if st.session_state.model is None:
        st.info("Train a model or load a checkpoint before evaluating.")
    else:
        if st.button("Evaluate model", type="primary"):
            try:
                with st.spinner("Evaluating…"):
                    _, test_loader = get_loaders(batch_size, train_limit, test_limit, augment, int(seed))
                    st.session_state.last_eval = run_epoch(st.session_state.model, test_loader, nn.CrossEntropyLoss(), device)
            except Exception as exc:
                st.error(f"Evaluation failed: {exc}")
        metrics = st.session_state.last_eval
        if metrics:
            a, b, c = st.columns(3)
            a.metric("Test accuracy", f"{metrics['accuracy']:.1%}")
            b.metric("Test loss", f"{metrics['loss']:.3f}")
            c.metric("Images evaluated", f"{int(metrics['examples']):,}")
            st.caption("Performance is an estimate on the selected test subset, not a guarantee on arbitrary photos.")

    st.divider()
    st.subheader("Load checkpoint")
    uploaded_checkpoint = st.file_uploader("Checkpoint (.pt or .pth)", type=["pt", "pth"], key="checkpoint-upload")
    if uploaded_checkpoint is not None and st.button("Load checkpoint"):
        try:
            payload_bytes = io.BytesIO(uploaded_checkpoint.getvalue())
            try:
                payload = torch.load(payload_bytes, map_location="cpu", weights_only=True)
            except TypeError:
                payload_bytes.seek(0)
                payload = torch.load(payload_bytes, map_location="cpu")
            if not isinstance(payload, dict) or "model_state_dict" not in payload:
                raise ValueError("This file is not a supported Neural Lab checkpoint.")
            model = CifarCNN()
            model.load_state_dict(payload["model_state_dict"])
            st.session_state.model = model.to(device)
            st.session_state.history = []
            st.session_state.last_eval = None
            st.success(f"Loaded checkpoint (epoch {payload.get('epoch', 'unknown')}).")
        except Exception as exc:
            st.error(f"Could not load checkpoint: {exc}")

with tab_predict:
    st.subheader("Try an image")
    st.write("Upload a picture to see which CIFAR-10 class the model considers most likely.")
    st.warning("This model only knows ten CIFAR-10 classes. A confident answer for an unrelated object can still be wrong.")
    uploaded_image = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg", "webp"], key="image-upload")
    if uploaded_image is not None:
        try:
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded image", width=280)
            if st.session_state.model is None:
                st.info("Train a model or load a checkpoint first.")
            elif st.button("Classify image", type="primary"):
                tensor = prepare_image(image).to(device)
                model = st.session_state.model
                model.eval()
                with torch.inference_mode():
                    probabilities = torch.softmax(model(tensor)[0], dim=0).cpu().numpy()
                top_indices = np.argsort(probabilities)[::-1][:5]
                best = int(top_indices[0])
                st.metric("Top prediction", CIFAR10_CLASSES[best], f"{probabilities[best]:.1%} model confidence")
                st.bar_chart({CIFAR10_CLASSES[int(i)]: float(probabilities[int(i)]) for i in top_indices})
                st.dataframe([{"Class": CIFAR10_CLASSES[int(i)], "Probability": float(probabilities[int(i)])} for i in top_indices],
                             use_container_width=True, hide_index=True)
        except (OSError, ValueError) as exc:
            st.error(f"Could not read that image: {exc}")

with tab_about:
    st.subheader("What is being trained?")
    st.markdown("""
**Architecture:** convolution, batch normalization, ReLU blocks, max-pooling, dropout,
adaptive average pooling, and a linear ten-class output layer.

**Loss:** cross-entropy. **Optimizer:** AdamW with weight decay.
**Preprocessing:** RGB conversion, 32×32 resize, and CIFAR-10 channel normalization.
Training augmentation uses random crop and horizontal flip.

### What the metrics mean
- **Training accuracy** measures performance on examples used for gradient updates.
- **Test accuracy** measures performance on held-out examples. Repeatedly tuning against a test split can make it less independent.
- **Loss** measures confidence-weighted classification error. Lower is generally better.

### Device notes
Auto-selection prefers NVIDIA CUDA, then Apple Silicon MPS, then CPU. MPS speed and behavior depend on installed PyTorch/macOS versions; CPU is the fallback.
""")
    st.code("Input:  N × 3 × 32 × 32\nOutput: N × 10 logits", language="text")
    st.caption("Neural Lab is an educational experiment, not a general-purpose object detector.")
